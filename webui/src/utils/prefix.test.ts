/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { proposePrefix } from "./prefix";

const PREFIX_PATTERN = /^[a-zA-Z0-9_-]+$/;

describe("proposePrefix", () => {
  it("lowercases and keeps alphanumerics", () => {
    expect(proposePrefix("MyServer")).toBe("myserver");
    expect(proposePrefix("Qwen2")).toBe("qwen2");
  });

  it("replaces every non-alphanumeric character with an underscore", () => {
    expect(proposePrefix("mcp/time")).toBe("mcp_time");
    expect(proposePrefix("@scope/server-name")).toBe("_scope_server_name");
    expect(proposePrefix("Qwen/Qwen2.5")).toBe("qwen_qwen2_5");
    expect(proposePrefix("a b")).toBe("a_b");
  });

  it("returns an empty string for empty input", () => {
    expect(proposePrefix("")).toBe("");
  });

  it("produces a value that satisfies the prefix criteria for non-empty ids", () => {
    for (const id of [
      "mcp/time",
      "@modelcontextprotocol/server-filesystem",
      "Qwen/Qwen2.5-7B",
      "my custom model!",
    ]) {
      expect(proposePrefix(id)).toMatch(PREFIX_PATTERN);
    }
  });
});
