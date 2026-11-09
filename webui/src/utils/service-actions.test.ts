/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import {
  CLOUD_DISABLED_HINT,
  LOAD_ERROR_HINT,
  getActionBlockReason,
} from "./service-actions";

const LOAD_ERROR = "Could not read the services configuration";

describe("getActionBlockReason", () => {
  it("returns the load error hint before any other reason", () => {
    const service = { is_cloud: true, disabled_reason: "No GPU found" };

    expect(getActionBlockReason(service, LOAD_ERROR, false)).toBe(
      LOAD_ERROR_HINT,
    );
  });

  it("blocks a local service when services.json failed to load", () => {
    const service = { is_cloud: false, disabled_reason: null };

    expect(getActionBlockReason(service, LOAD_ERROR, true)).toBe(
      LOAD_ERROR_HINT,
    );
  });

  it("returns the cloud reason before the hardware reason when cloud is disabled", () => {
    const service = { is_cloud: true, disabled_reason: "No GPU found" };

    expect(getActionBlockReason(service, null, false)).toBe(
      CLOUD_DISABLED_HINT,
    );
  });

  it("does not block a cloud service when the cloud setting is unknown", () => {
    const service = { is_cloud: true, disabled_reason: null };

    expect(getActionBlockReason(service, null, undefined)).toBeNull();
  });

  it("does not block a local service when cloud is disabled", () => {
    const service = { is_cloud: false, disabled_reason: null };

    expect(getActionBlockReason(service, null, false)).toBeNull();
  });

  it("returns the hardware reason when nothing else blocks the service", () => {
    const service = { is_cloud: false, disabled_reason: "No GPU found" };

    expect(getActionBlockReason(service, undefined, true)).toBe("No GPU found");
  });

  it("returns null when nothing blocks the service", () => {
    const service = { is_cloud: false, disabled_reason: "" };

    expect(getActionBlockReason(service, null, true)).toBeNull();
  });
});
