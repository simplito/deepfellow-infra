/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, it, expect } from "vitest";
import {
  clearModelInstallProgress,
  clearServiceInstallProgress,
  getSnapshot,
  setModelInstallProgress,
  setServiceInstallProgress,
} from "./install-progress-store";

// Each test uses a unique ID to avoid cross-test state pollution
// (the store is module-level and persists between tests in the same file).
let _id = 0;
function uid(): string {
  return `test-${_id++}`;
}

// ── Services ──────────────────────────────────────────────────────────────────

describe("setServiceInstallProgress", () => {
  it("stores the given progress", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "download", value: 0.4 });
    expect(getSnapshot().services[id]).toEqual({ stage: "download", value: 0.4 });
  });

  // ── Bug 2 / monotonicity — value regression ───────────────────────────────
  // The cleanup function used to call clearServiceInstallProgress before the
  // next effect run had a chance to re-populate the store, causing the progress
  // bar to flash back to the stale (lower) value from the backend cache.
  // The store defends against this by rejecting value decreases within a stage.
  describe("value monotonicity (Bug 2 — no regression when effect cleanup races React Query refetch)", () => {
    it("ignores a lower value within the same stage", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.6 });
      setServiceInstallProgress(id, { stage: "download", value: 0.3 }); // stale — must be ignored
      expect(getSnapshot().services[id]?.value).toBe(0.6);
    });

    it("accepts a higher value within the same stage", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.3 });
      setServiceInstallProgress(id, { stage: "download", value: 0.7 });
      expect(getSnapshot().services[id]?.value).toBe(0.7);
    });

    it("accepts the same value (no-op, no error)", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.5 });
      setServiceInstallProgress(id, { stage: "download", value: 0.5 });
      expect(getSnapshot().services[id]?.value).toBe(0.5);
    });
  });

  // ── Stage monotonicity ────────────────────────────────────────────────────
  describe("stage monotonicity (download → install is one-way)", () => {
    it("allows advancing from download to install", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.9 });
      setServiceInstallProgress(id, { stage: "install", value: 0.1 });
      expect(getSnapshot().services[id]?.stage).toBe("install");
    });

    it("rejects reverting from install back to download", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "install", value: 0.3 });
      setServiceInstallProgress(id, { stage: "download", value: 0.9 }); // regression — must be ignored
      expect(getSnapshot().services[id]?.stage).toBe("install");
    });

    it("allows value advancement after a stage change", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 1.0 });
      setServiceInstallProgress(id, { stage: "install", value: 0.5 });
      setServiceInstallProgress(id, { stage: "install", value: 0.8 });
      expect(getSnapshot().services[id]).toEqual({ stage: "install", value: 0.8 });
    });
  });

  // ── Bug 5 — concurrent progress sources ───────────────────────────────────
  // When the 10 s refetch fires mid-install, the polling effect may start a
  // second simulation for the same service alongside the active mutation.
  // The store's monotonicity rules ensure whichever source is ahead wins and
  // neither can roll back the bar.
  describe("concurrent progress sources (Bug 5 — double-tracking safety net)", () => {
    it("whichever source is ahead wins — slower source cannot roll back progress", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.5 }); // mutation
      setServiceInstallProgress(id, { stage: "download", value: 0.2 }); // polling effect (lagging)
      expect(getSnapshot().services[id]?.value).toBe(0.5);
    });

    it("the lagging source eventually catches up once it exceeds the current value", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 0.5 });
      setServiceInstallProgress(id, { stage: "download", value: 0.8 }); // lagging source caught up
      expect(getSnapshot().services[id]?.value).toBe(0.8);
    });
  });

  // ── Value clamping ────────────────────────────────────────────────────────
  describe("value clamping", () => {
    it("clamps values above 1 to 1", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: 1.5 });
      expect(getSnapshot().services[id]?.value).toBe(1);
    });

    it("clamps negative values to 0", () => {
      const id = uid();
      setServiceInstallProgress(id, { stage: "download", value: -0.1 });
      expect(getSnapshot().services[id]?.value).toBe(0);
    });
  });
});

describe("clearServiceInstallProgress", () => {
  it("removes the entry from the store", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "download", value: 0.5 });
    clearServiceInstallProgress(id);
    expect(getSnapshot().services[id]).toBeUndefined();
  });

  // After clear the store must accept any fresh value — this is the path taken
  // on page reload: the reconciliation step clears finished services, then the
  // polling effect re-populates from the backend value (which may be lower than
  // where the simulation last was before reload).
  it("after clear, accepts a new lower value (simulates page-reload re-sync from backend)", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "install", value: 0.9 });
    clearServiceInstallProgress(id);
    setServiceInstallProgress(id, { stage: "download", value: 0.4 }); // backend reported 40%
    expect(getSnapshot().services[id]).toEqual({ stage: "download", value: 0.4 });
  });

  it("is a no-op when the entry does not exist", () => {
    const id = uid();
    expect(() => clearServiceInstallProgress(id)).not.toThrow();
    expect(getSnapshot().services[id]).toBeUndefined();
  });
});

// ── Models ────────────────────────────────────────────────────────────────────

describe("setModelInstallProgress", () => {
  it("stores progress under a serviceId::modelId composite key", () => {
    const svc = uid();
    const mdl = uid();
    setModelInstallProgress(svc, mdl, { stage: "download", value: 0.35 });
    expect(getSnapshot().models[`${svc}::${mdl}`]).toEqual({ stage: "download", value: 0.35 });
  });

  it("tracks multiple models independently", () => {
    const svc = uid();
    const mdl1 = uid();
    const mdl2 = uid();
    setModelInstallProgress(svc, mdl1, { stage: "download", value: 0.2 });
    setModelInstallProgress(svc, mdl2, { stage: "install", value: 0.6 });
    expect(getSnapshot().models[`${svc}::${mdl1}`]?.value).toBe(0.2);
    expect(getSnapshot().models[`${svc}::${mdl2}`]?.stage).toBe("install");
  });

  it("applies the same monotonicity rules as service progress", () => {
    const svc = uid();
    const mdl = uid();
    setModelInstallProgress(svc, mdl, { stage: "download", value: 0.7 });
    setModelInstallProgress(svc, mdl, { stage: "download", value: 0.3 }); // regression — ignored
    expect(getSnapshot().models[`${svc}::${mdl}`]?.value).toBe(0.7);
  });
});

describe("clearModelInstallProgress", () => {
  it("removes only the targeted model, leaving others intact", () => {
    const svc = uid();
    const mdl1 = uid();
    const mdl2 = uid();
    setModelInstallProgress(svc, mdl1, { stage: "download", value: 0.4 });
    setModelInstallProgress(svc, mdl2, { stage: "download", value: 0.6 });
    clearModelInstallProgress(svc, mdl1);
    expect(getSnapshot().models[`${svc}::${mdl1}`]).toBeUndefined();
    expect(getSnapshot().models[`${svc}::${mdl2}`]?.value).toBe(0.6);
  });

  it("after clear, accepts a fresh lower value (page-reload re-sync path)", () => {
    const svc = uid();
    const mdl = uid();
    setModelInstallProgress(svc, mdl, { stage: "install", value: 0.95 });
    clearModelInstallProgress(svc, mdl);
    setModelInstallProgress(svc, mdl, { stage: "download", value: 0.5 });
    expect(getSnapshot().models[`${svc}::${mdl}`]).toEqual({ stage: "download", value: 0.5 });
  });
});

// ── Bug 3 guard ───────────────────────────────────────────────────────────────
// Previously hasRealProgressRef was only set for 0 < value < 1, leaving the
// boundary values (0 and 1) able to trigger unnecessary fake-data toast updates.
// The store itself is unaffected; this suite documents the boundary behaviour
// so a future regression in the component flag logic is at least noticed here.
describe("boundary values 0 and 1 are accepted (Bug 3 — hasRealProgressRef must be set for any SSE value)", () => {
  it("accepts value 0 as valid initial progress", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "download", value: 0 });
    expect(getSnapshot().services[id]).toEqual({ stage: "download", value: 0 });
  });

  it("accepts value 1 as valid completion signal", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "install", value: 1 });
    expect(getSnapshot().services[id]).toEqual({ stage: "install", value: 1 });
  });

  it("rejects a lower value after 1 is set", () => {
    const id = uid();
    setServiceInstallProgress(id, { stage: "install", value: 1 });
    setServiceInstallProgress(id, { stage: "install", value: 0.5 });
    expect(getSnapshot().services[id]?.value).toBe(1);
  });
});
