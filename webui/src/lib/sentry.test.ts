/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { ErrorEvent } from "@sentry/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getApiKey, sentryInit } = vi.hoisted(() => ({
  getApiKey: vi.fn<() => string | null>(),
  sentryInit: vi.fn(),
}));

vi.mock("@/deepfellow/client", () => ({
  AdminApiKeyStorage: { get: getApiKey },
}));
vi.mock("@sentry/react", () => ({ init: sentryInit }));

import {
  SENTRY_CONFIG_PATH,
  SENTRY_TUNNEL_PATH,
  type SentryBrowserConfig,
  buildSentryOptions,
  fetchSentryConfig,
  initSentry,
  scrubEvent,
  withoutSessions,
} from "./sentry";

const DISABLED: SentryBrowserConfig = {
  enabled: false,
  dsn: null,
  environment: null,
  release: null,
};

const ENABLED: SentryBrowserConfig = {
  enabled: true,
  dsn: "https://public-key@o1.ingest.sentry.io/123",
  environment: "staging",
  release: "1.2.3",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

function fetchReturning(response: Response) {
  return vi.fn<typeof fetch>().mockResolvedValue(response);
}

beforeEach(() => {
  getApiKey.mockReturnValue(null);
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe("fetchSentryConfig", () => {
  it("returns the config from the public endpoint without credentials", async () => {
    const fetchImpl = fetchReturning(jsonResponse(ENABLED));

    await expect(fetchSentryConfig(fetchImpl)).resolves.toEqual(ENABLED);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(SENTRY_CONFIG_PATH);
    expect(init?.headers).toBeUndefined();
    expect(init?.signal).toBeInstanceOf(AbortSignal);
  });

  it("accepts the disabled config, which has no release", async () => {
    await expect(
      fetchSentryConfig(fetchReturning(jsonResponse(DISABLED))),
    ).resolves.toEqual(DISABLED);
  });

  it("returns null on a non-OK response", async () => {
    const fetchImpl = fetchReturning(jsonResponse({ detail: "boom" }, 500));

    await expect(fetchSentryConfig(fetchImpl)).resolves.toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it("returns null when the request fails", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(fetchSentryConfig(fetchImpl)).resolves.toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it("returns null when the request times out", async () => {
    const fetchImpl = vi.fn<typeof fetch>(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(init.signal?.reason),
          );
        }),
    );

    await expect(fetchSentryConfig(fetchImpl, 10)).resolves.toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it("returns null when the body isn't JSON", async () => {
    const fetchImpl = fetchReturning(new Response("<html>", { status: 200 }));

    await expect(fetchSentryConfig(fetchImpl)).resolves.toBeNull();
  });

  it.each([
    ["null", null],
    ["a string", "enabled"],
    ["a missing release", { enabled: true, dsn: "x", environment: "e" }],
    ["a non-boolean enabled", { ...ENABLED, enabled: "true" }],
    ["a numeric dsn", { ...ENABLED, dsn: 1 }],
  ])("returns null for %s", async (_label, body) => {
    const fetchImpl = fetchReturning(jsonResponse(body));

    await expect(fetchSentryConfig(fetchImpl)).resolves.toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });
});

describe("buildSentryOptions", () => {
  it("builds error-only options from an enabled config", () => {
    const options = buildSentryOptions(ENABLED);

    expect(options).toEqual({
      dsn: ENABLED.dsn,
      environment: "staging",
      release: "1.2.3",
      tunnel: SENTRY_TUNNEL_PATH,
      integrations: withoutSessions,
      dataCollection: {
        userInfo: false,
        cookies: false,
        httpHeaders: false,
        httpBodies: [],
        urlQueryParams: false,
      },
      beforeSend: scrubEvent,
    });
  });

  it("returns null when Sentry is disabled", () => {
    expect(buildSentryOptions(DISABLED)).toBeNull();
  });

  it("sends events through the same-origin tunnel", () => {
    // The CSP allows `connect-src 'self'` only, so events sent straight to the DSN host are blocked.
    const tunnel = buildSentryOptions(ENABLED)?.tunnel;

    expect(tunnel).toBe("/admin/config/sentry/tunnel");
    expect(new URL(tunnel ?? "", "https://infra.example").origin).toBe(
      "https://infra.example",
    );
  });

  it("leaves the release to the SDK when there is none", () => {
    expect(
      buildSentryOptions({ ...ENABLED, release: null })?.release,
    ).toBeUndefined();
  });

  it("returns null when enabled without a DSN", () => {
    expect(buildSentryOptions({ ...ENABLED, dsn: null })).toBeNull();
    expect(buildSentryOptions({ ...ENABLED, dsn: "" })).toBeNull();
  });

  it("leaves the environment to the SDK when there is none", () => {
    expect(
      buildSentryOptions({ ...ENABLED, environment: null })?.environment,
    ).toBeUndefined();
  });
});

describe("withoutSessions", () => {
  it("drops only the session integration", () => {
    const defaults = [
      { name: "BrowserSession" },
      { name: "GlobalHandlers" },
      { name: "Dedupe" },
    ];

    expect(withoutSessions(defaults).map((i) => i.name)).toEqual([
      "GlobalHandlers",
      "Dedupe",
    ]);
  });
});

describe("scrubEvent", () => {
  it("replaces the admin API key everywhere in the event", () => {
    getApiKey.mockReturnValue("sk-admin");
    const event = {
      message: "failed with sk-admin",
      extra: { nested: ["Bearer sk-admin", 42, null], sk_admin: "sk-admin" },
    } as unknown as ErrorEvent;

    expect(scrubEvent(event)).toEqual({
      message: "failed with [Filtered]",
      extra: {
        nested: ["Bearer [Filtered]", 42, null],
        sk_admin: "[Filtered]",
      },
    });
  });

  it("returns the event unchanged when no key is stored", () => {
    const event = { message: "plain" } as ErrorEvent;

    expect(scrubEvent(event)).toBe(event);
  });

  it("drops the event when scrubbing fails", () => {
    getApiKey.mockImplementation(() => {
      throw new Error("storage unavailable");
    });

    expect(scrubEvent({ message: "x" } as ErrorEvent)).toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });
});

describe("initSentry", () => {
  it("initializes Sentry when enabled", async () => {
    await initSentry(fetchReturning(jsonResponse(ENABLED)));

    expect(sentryInit).toHaveBeenCalledTimes(1);
    expect(sentryInit).toHaveBeenCalledWith(buildSentryOptions(ENABLED));
  });

  it("doesn't initialize Sentry when disabled", async () => {
    await initSentry(fetchReturning(jsonResponse(DISABLED)));

    expect(sentryInit).not.toHaveBeenCalled();
  });

  it("resolves without initializing when fetching the config fails", async () => {
    await expect(
      initSentry(vi.fn<typeof fetch>().mockRejectedValue(new Error("down"))),
    ).resolves.toBeUndefined();

    expect(sentryInit).not.toHaveBeenCalled();
  });

  it("resolves when Sentry.init throws", async () => {
    sentryInit.mockImplementationOnce(() => {
      throw new Error("bad init");
    });

    await expect(
      initSentry(fetchReturning(jsonResponse(ENABLED))),
    ).resolves.toBeUndefined();
    expect(console.warn).toHaveBeenCalledTimes(1);
  });
});
