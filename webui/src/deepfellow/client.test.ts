/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DeepFellowClient, formatApiErrorMessage } from "./client";

const FALLBACK = "<raw body>";

// The body the server's exception handlers send (server/error_handlers.py).
function openAiBody(
  message: string,
  type = "api_error",
  param: string | null = null,
) {
  return { error: { message, type, param, code: null } };
}

describe("formatApiErrorMessage", () => {
  describe("OpenAI-style bodies", () => {
    it("returns the message of the 503 guard", () => {
      const message =
        "Could not read the services configuration (...). Fix or restore the file, then restart DF Infra.";

      expect(formatApiErrorMessage(openAiBody(message), FALLBACK)).toBe(
        message,
      );
    });

    it("returns the message of an HTTPException such as a 405", () => {
      const body = openAiBody(
        "Action not supported by this service type",
        "invalid_request_error",
      );

      expect(formatApiErrorMessage(body, FALLBACK)).toBe(
        "Action not supported by this service type",
      );
    });

    it("returns the message alone for a validation error, not its param", () => {
      const body = openAiBody(
        "Field required",
        "invalid_request_error",
        "model",
      );

      expect(formatApiErrorMessage(body, FALLBACK)).toBe("Field required");
    });

    it("returns the message of an unhandled 500 with its error ID", () => {
      const message =
        "Internal server error. Error ID: c4d89da4-df28-49d1-b88a-2299e33aff1a";

      expect(formatApiErrorMessage(openAiBody(message), FALLBACK)).toBe(
        message,
      );
    });

    it("prefers error.message over detail and the top-level message", () => {
      const body = {
        error: { message: "from error" },
        detail: "from detail",
        message: "from message",
      };

      expect(formatApiErrorMessage(body, FALLBACK)).toBe("from error");
    });
  });

  describe("other body shapes", () => {
    it("returns a plain detail string", () => {
      const body = {
        detail: "Cannot update installed server. Uninstall first.",
      };

      expect(formatApiErrorMessage(body, FALLBACK)).toBe(
        "Cannot update installed server. Uninstall first.",
      );
    });

    it("joins a detail list, prefixing the field when there is one", () => {
      const body = {
        detail: [
          { field: "model", message: "field required" },
          { msg: "bad value" },
        ],
      };

      expect(formatApiErrorMessage(body, FALLBACK)).toBe(
        "model: field required; bad value",
      );
    });

    it("joins a detail list of strings", () => {
      expect(
        formatApiErrorMessage({ detail: ["first", "second"] }, FALLBACK),
      ).toBe("first; second");
    });

    it("stringifies a detail object", () => {
      expect(
        formatApiErrorMessage({ detail: { warnings: ["w1"] } }, FALLBACK),
      ).toBe('{"warnings":["w1"]}');
    });

    it("returns a top-level message", () => {
      expect(formatApiErrorMessage({ message: "Throttled" }, FALLBACK)).toBe(
        "Throttled",
      );
    });

    it("prefers detail over the top-level message", () => {
      expect(
        formatApiErrorMessage(
          { detail: "from detail", message: "from message" },
          FALLBACK,
        ),
      ).toBe("from detail");
    });
  });

  describe("nothing readable", () => {
    it.each([
      ["an empty error.message", { error: { message: "" } }],
      ["a blank error.message", { error: { message: "   " } }],
      ["a non-string error.message", { error: { message: 42 } }],
      ["an error without a message", { error: { type: "api_error" } }],
      ["an error that is a bare string", { error: "boom" }],
      ["a null error", { error: null }],
      ["no known field", { foo: "bar" }],
      ["an empty object", {}],
    ])("returns the fallback for %s", (_name, body) => {
      expect(formatApiErrorMessage(body, FALLBACK)).toBe(FALLBACK);
    });

    it.each([
      ["a string", "boom"],
      ["a number", 42],
      ["null", null],
      ["an array", []],
    ])("returns the fallback for a body that is %s", (_name, body) => {
      expect(formatApiErrorMessage(body, FALLBACK)).toBe(FALLBACK);
    });
  });
});

describe("API client error text", () => {
  const MESSAGE = "Could not read the services configuration";

  // Each request path that reads an error body, called with the minimum arguments.
  const requestPaths: [
    string,
    (client: DeepFellowClient) => Promise<unknown>,
  ][] = [
    ["plain request", (client) => client.getAdminService("svc")],
    [
      "service install stream",
      (client) => client.installAdminServiceStreaming("svc", {}, () => {}),
    ],
    [
      "model install stream",
      (client) =>
        client.installAdminServiceModelStreaming("svc", "model", {}, () => {}),
    ],
    ["catalog refresh", (client) => client.refreshCatalog("svc")],
  ];

  function stubFetch(status: number, content: string, statusText = "") {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(content, { status, statusText })),
    );
  }

  beforeEach(() => {
    // AdminApiKeyStorage reads localStorage, which the node test environment lacks.
    vi.stubGlobal("localStorage", {
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe.each(requestPaths)("%s", (_name, call) => {
    it("shows the message of an OpenAI-style 503, not the JSON", async () => {
      stubFetch(503, JSON.stringify(openAiBody(MESSAGE)));

      await expect(call(new DeepFellowClient())).rejects.toMatchObject({
        message: `HTTP 503: ${MESSAGE}`,
      });
    });
  });

  it("keeps the HTTP status prefix on a 404", async () => {
    stubFetch(
      404,
      JSON.stringify(openAiBody("Service not found", "not_found_error")),
    );

    await expect(
      new DeepFellowClient().getAdminService("svc"),
    ).rejects.toMatchObject({
      message: "HTTP 404: Service not found",
    });
  });

  it("shows a non-JSON body as received", async () => {
    stubFetch(502, "<html>Bad Gateway</html>");

    await expect(
      new DeepFellowClient().getAdminService("svc"),
    ).rejects.toMatchObject({
      message: "HTTP 502: <html>Bad Gateway</html>",
    });
  });

  it("shows the status text when the body is empty", async () => {
    stubFetch(504, "", "Gateway Timeout");

    await expect(
      new DeepFellowClient().getAdminService("svc"),
    ).rejects.toMatchObject({
      message: "HTTP 504: Gateway Timeout",
    });
  });

  it("shows the body unchanged when its error message is empty", async () => {
    const content = JSON.stringify({
      error: { message: "", type: "api_error" },
    });
    stubFetch(500, content);

    await expect(
      new DeepFellowClient().getAdminService("svc"),
    ).rejects.toMatchObject({
      message: `HTTP 500: ${content}`,
    });
  });
});
