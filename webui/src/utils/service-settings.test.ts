import type { SpecField } from "@/deepfellow/types";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { buildSettingEntries, formatSettingValue } from "./service-settings";

function field(
  partial: Partial<SpecField> & { name: string; type: SpecField["type"] },
): SpecField {
  return {
    description: partial.name,
    required: false,
    ...partial,
  };
}

describe("formatSettingValue", () => {
  it("returns a dash for null/undefined", () => {
    expect(formatSettingValue(null)).toBe("—");
    expect(formatSettingValue(undefined)).toBe("—");
  });

  it("maps a oneof value to its label", () => {
    const f = field({
      name: "hardware",
      type: "oneof",
      values: [{ label: "GPU (CUDA)", value: "cuda" }, "cpu"],
    });
    expect(formatSettingValue("cuda", f)).toBe("GPU (CUDA)");
    expect(formatSettingValue("cpu", f)).toBe("cpu");
    expect(formatSettingValue("unknown", f)).toBe("unknown");
  });

  it("renders booleans as Yes/No", () => {
    const f = field({ name: "enabled", type: "bool" });
    expect(formatSettingValue(true, f)).toBe("Yes");
    expect(formatSettingValue("true", f)).toBe("Yes");
    expect(formatSettingValue(false, f)).toBe("No");
  });

  it("joins lists and shows a dash when empty", () => {
    const f = field({ name: "args", type: "list" });
    expect(formatSettingValue(["a", "b"], f)).toBe("a, b");
    expect(formatSettingValue([], f)).toBe("—");
  });

  it("formats maps as key: value pairs", () => {
    const f = field({ name: "env", type: "map" });
    expect(formatSettingValue({ A: "1", B: "2" }, f)).toBe("A: 1, B: 2");
    expect(formatSettingValue({}, f)).toBe("—");
  });

  it("stringifies plain values and unknown objects", () => {
    expect(
      formatSettingValue("https://x", field({ name: "url", type: "text" })),
    ).toBe("https://x");
    expect(
      formatSettingValue(42, field({ name: "port", type: "number" })),
    ).toBe("42");
    expect(formatSettingValue({ a: 1 })).toBe('{"a":1}');
  });
});

describe("buildSettingEntries", () => {
  const fields: SpecField[] = [
    field({ name: "base_url", type: "text", description: "Base URL" }),
    field({ name: "api_key", type: "password", description: "API Key" }),
    field({ name: "missing", type: "text", description: "Not provided" }),
  ];

  it("pairs installed values with field metadata in spec order", () => {
    const entries = buildSettingEntries(fields, {
      api_key: "secret",
      base_url: "https://x",
    });
    expect(entries.map((e) => e.key)).toEqual(["base_url", "api_key"]);
    expect(entries[0].label).toBe("Base URL");
    expect(entries[0].isSecret).toBe(false);
    expect(entries[1].label).toBe("API Key");
    expect(entries[1].isSecret).toBe(true);
  });

  it("skips spec fields that have no installed value", () => {
    const entries = buildSettingEntries(fields, { base_url: "https://x" });
    expect(entries.map((e) => e.key)).toEqual(["base_url"]);
  });

  it("appends extra keys not present in the spec, using the raw key as label", () => {
    const entries = buildSettingEntries(fields, {
      base_url: "https://x",
      extra: "value",
    });
    expect(entries.map((e) => e.key)).toEqual(["base_url", "extra"]);
    const extra = entries[1];
    expect(extra.label).toBe("extra");
    expect(extra.field).toBeUndefined();
    expect(extra.isSecret).toBe(false);
  });
});
