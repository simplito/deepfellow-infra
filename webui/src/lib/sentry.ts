/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { AdminApiKeyStorage } from "@/deepfellow/client";
import * as Sentry from "@sentry/react";
import type { BrowserOptions, ErrorEvent } from "@sentry/react";

export const SENTRY_CONFIG_PATH = "/admin/config/sentry";
// Events go through Infra: the WebUI's CSP (`connect-src 'self'`) blocks requests to the Sentry host.
// A relative path, so it is same-origin wherever the WebUI is served from.
export const SENTRY_TUNNEL_PATH = "/admin/config/sentry/tunnel";
// Startup waits for the config at most this long, so an unreachable backend can't hold up the UI.
export const SENTRY_CONFIG_TIMEOUT_MS = 3000;

const FILTERED = "[Filtered]";

type Integration = Extract<BrowserOptions["integrations"], unknown[]>[number];

const NO_DATA_COLLECTION: NonNullable<BrowserOptions["dataCollection"]> = {
  userInfo: false,
  cookies: false,
  httpHeaders: false,
  httpBodies: [],
  urlQueryParams: false,
};

export interface SentryBrowserConfig {
  enabled: boolean;
  dsn: string | null;
  environment: string | null;
  // null while Sentry is off: the backend doesn't reveal its version then.
  release: string | null;
}

function isSentryBrowserConfig(value: unknown): value is SentryBrowserConfig {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.enabled === "boolean" &&
    (v.dsn === null || typeof v.dsn === "string") &&
    (v.environment === null || typeof v.environment === "string") &&
    (v.release === null || typeof v.release === "string")
  );
}

/** Fetch the browser Sentry config from the backend; never throws, returns null when it can't be read. */
export async function fetchSentryConfig(
  fetchImpl: typeof fetch = fetch,
  timeoutMs = SENTRY_CONFIG_TIMEOUT_MS,
): Promise<SentryBrowserConfig | null> {
  try {
    const response = await fetchImpl(SENTRY_CONFIG_PATH, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      console.warn(
        `Sentry is off: fetching its config failed (HTTP ${response.status}).`,
      );
      return null;
    }
    const body: unknown = await response.json();
    if (!isSentryBrowserConfig(body)) {
      console.warn("Sentry is off: its config has an unexpected shape.");
      return null;
    }
    return body;
  } catch (e) {
    console.warn("Sentry is off: fetching its config failed.", e);
    return null;
  }
}

function scrub(value: unknown, secrets: string[]): unknown {
  if (typeof value === "string") {
    return secrets.reduce((s, secret) => s.split(secret).join(FILTERED), value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => scrub(item, secrets));
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, scrub(item, secrets)]),
    );
  }
  return value;
}

/**
 * Replace every occurrence of the stored admin API key in `event` with "[Filtered]".
 * If scrubbing fails the event is dropped, never sent unscrubbed.
 */
export function scrubEvent(event: ErrorEvent): ErrorEvent | null {
  try {
    const apiKey = AdminApiKeyStorage.get();
    return apiKey ? (scrub(event, [apiKey]) as ErrorEvent) : event;
  } catch (e) {
    console.warn("Dropped a Sentry event because scrubbing it failed.", e);
    return null;
  }
}

/** Drop the default session tracking: it would send an envelope on every page load, and only errors are reported. */
export function withoutSessions(defaults: Integration[]): Integration[] {
  return defaults.filter(
    (integration) => integration.name !== "BrowserSession",
  );
}

/** Build the SDK options from `config`, or null when Sentry is off. Error capture only: no tracing, no replay, no request data. */
export function buildSentryOptions(
  config: SentryBrowserConfig,
): BrowserOptions | null {
  if (!config.enabled || !config.dsn) return null;
  return {
    dsn: config.dsn,
    environment: config.environment ?? undefined,
    release: config.release ?? undefined,
    tunnel: SENTRY_TUNNEL_PATH,
    integrations: withoutSessions,
    // Nothing beyond the error itself: no user info, cookies, headers (the admin API key travels
    // in Authorization), bodies or query params.
    dataCollection: NO_DATA_COLLECTION,
    beforeSend: scrubEvent,
  };
}

/** Initialize Sentry from the backend config; never throws, so a failure can't block startup. */
export async function initSentry(
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  try {
    const config = await fetchSentryConfig(fetchImpl);
    const options = config && buildSentryOptions(config);
    if (options) Sentry.init(options);
  } catch (e) {
    console.warn("Sentry could not be initialized.", e);
  }
}
