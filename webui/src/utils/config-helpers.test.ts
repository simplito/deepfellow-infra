import type { ConfigEntry } from "@/deepfellow/types";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { isBooleanValue, sortConfigEntries } from "./config-helpers";

describe("isBooleanValue", () => {
  it.each(["True", "False"])(
    "treats Python's str(bool) %s as boolean",
    (value) => {
      expect(isBooleanValue(value)).toBe(true);
    },
  );

  it.each(["", "true", "false", "1", "0", "some-string"])(
    "treats %j as a non-boolean value",
    (value) => {
      expect(isBooleanValue(value)).toBe(false);
    },
  );
});

function entry(key: string, value: string, is_secret = false): ConfigEntry {
  return {
    key,
    value,
    is_secret,
    field_name: key.replace(/^DF_/, "").toLowerCase(),
    is_editable: true,
  };
}

describe("sortConfigEntries", () => {
  it("puts filled entries and secrets before empty ones, keeping order otherwise", () => {
    const entries = [
      entry("DF_A", ""),
      entry("DF_B", "b"),
      entry("DF_C", "", true),
      entry("DF_D", ""),
      entry("DF_E", "e"),
    ];

    expect(sortConfigEntries(entries).map((e) => e.key)).toEqual([
      "DF_B",
      "DF_C",
      "DF_E",
      "DF_A",
      "DF_D",
    ]);
  });

  it("keeps OTel and Sentry entries together when some are empty", () => {
    const entries = [
      entry("DF_EMPTY", ""),
      entry("DF_NAME", "infra"),
      entry("DF_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
      entry("DF_OTEL_TRACING_ENABLED", "False"),
      entry("DF_SENTRY_ENABLED", "False"),
      entry("DF_SENTRY_DSN", "••••••••", true),
      entry("DF_SENTRY_ENVIRONMENT", ""),
      entry("DF_LAST", "x"),
    ];

    expect(sortConfigEntries(entries).map((e) => e.key)).toEqual([
      "DF_NAME",
      "DF_OTEL_EXPORTER_OTLP_ENDPOINT",
      "DF_OTEL_TRACING_ENABLED",
      "DF_SENTRY_ENABLED",
      "DF_SENTRY_DSN",
      "DF_SENTRY_ENVIRONMENT",
      "DF_LAST",
      "DF_EMPTY",
    ]);
  });

  it("sorts the observability group as empty when all its entries are", () => {
    const entries = [entry("DF_SENTRY_ENVIRONMENT", ""), entry("DF_NAME", "x")];

    expect(sortConfigEntries(entries).map((e) => e.key)).toEqual([
      "DF_NAME",
      "DF_SENTRY_ENVIRONMENT",
    ]);
  });

  it("doesn't mutate its input", () => {
    const entries = [entry("DF_A", ""), entry("DF_B", "b")];

    sortConfigEntries(entries);

    expect(entries.map((e) => e.key)).toEqual(["DF_A", "DF_B"]);
  });
});
