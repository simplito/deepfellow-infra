/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { startProgressSimulation } from "./progress-simulation";

const TICK_MS = 50;

describe("startProgressSimulation", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  // ── Bug 1 ─────────────────────────────────────────────────────────────────
  // On page reload the backend already reports a non-zero progress value.
  // The simulation must start from that value so the bar does not flash to 0%.
  describe("initialValue (Bug 1 — no reset to 0% on page reload)", () => {
    it("starts from 0 by default when no initialValue is given", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.1,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS);
      sim.stop();

      expect(ticks).toHaveLength(1);
      expect(ticks[0]).toBeGreaterThan(0);
      expect(ticks[0]).toBeLessThan(0.15);
    });

    it("starts from initialValue and never drops below it on the first tick", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.001, // very slow — ensures the first tick value is driven by initialValue
        initialValue: 0.55,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS);
      sim.stop();

      expect(ticks[0]).toBeGreaterThanOrEqual(0.55);
    });

    it("never produces a value below initialValue across many ticks", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.001,
        initialValue: 0.7,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS * 20);
      sim.stop();

      expect(ticks.length).toBeGreaterThan(0);
      expect(Math.min(...ticks)).toBeGreaterThanOrEqual(0.7);
    });

    it("clamps initialValue above 1 to the MAX_SIMULATED ceiling", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.1,
        initialValue: 1.5,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS);
      sim.stop();

      expect(ticks[0]).toBeLessThanOrEqual(1);
    });

    it("clamps negative initialValue to 0", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.1,
        initialValue: -0.3,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS);
      sim.stop();

      expect(ticks[0]).toBeGreaterThanOrEqual(0);
    });
  });

  // ── Bug 4 ─────────────────────────────────────────────────────────────────
  // If real SSE data has advanced the store to e.g. 90% while the simulation's
  // internal counter is at 40%, smoothComplete must sync to 90% before computing
  // the animation — otherwise the bar freezes for several seconds.
  describe("smoothComplete with currentValue (Bug 4 — no freeze when real data is ahead)", () => {
    it("animates from currentValue when it is ahead of simulation's internal position", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.001, // very slow so simulation stays near 0
        initialValue: 0,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS * 4); // ~4 ticks, simulation ≈ 0.004
      ticks.length = 0;

      // Real SSE data pushed the store to 0.9 — pass it as currentValue
      sim.smoothComplete(5000, undefined, 0.9);
      vi.advanceTimersByTime(TICK_MS); // first smooth tick

      // Must start from 0.9, not from ~0.004
      expect(ticks[0]).toBeGreaterThanOrEqual(0.9);
      sim.stop();
    });

    it("does not roll back when currentValue is behind simulation's position", () => {
      const ticks: number[] = [];
      const sim = startProgressSimulation({
        stepPerTick: 0.2,
        initialValue: 0.8,
        onTick: (v) => ticks.push(v),
      });
      vi.advanceTimersByTime(TICK_MS * 4); // simulation well past 0.8
      ticks.length = 0;

      // Stale store value — must be ignored
      sim.smoothComplete(5000, undefined, 0.3);
      vi.advanceTimersByTime(TICK_MS);

      expect(ticks[0]).toBeGreaterThan(0.3);
      sim.stop();
    });

    it("calls onDone when the animation reaches 1.0", () => {
      let done = false;
      const sim = startProgressSimulation({
        stepPerTick: 0.001,
        initialValue: 0,
        onTick: () => {},
      });
      // Pass currentValue near 1 so smoothComplete completes quickly
      sim.smoothComplete(TICK_MS * 2, () => { done = true; }, 0.999);
      vi.advanceTimersByTime(TICK_MS * 10);

      expect(done).toBe(true);
    });

    it("calls onDone immediately when simulation is already at 1.0", () => {
      let done = false;
      const sim = startProgressSimulation({
        stepPerTick: 0.001,
        initialValue: 1,
        onTick: () => {},
      });
      sim.smoothComplete(5000, () => { done = true; }, 1);
      vi.advanceTimersByTime(0); // synchronous completion

      expect(done).toBe(true);
    });
  });

  // ── General behaviour ─────────────────────────────────────────────────────
  it("stop() halts all ticks immediately", () => {
    const ticks: number[] = [];
    const sim = startProgressSimulation({ stepPerTick: 0.1, onTick: (v) => ticks.push(v) });
    vi.advanceTimersByTime(TICK_MS * 4);
    const countBefore = ticks.length;
    sim.stop();
    vi.advanceTimersByTime(TICK_MS * 10);

    expect(ticks.length).toBe(countBefore);
  });

  it("value never exceeds 1.0 during normal progression", () => {
    const ticks: number[] = [];
    const sim = startProgressSimulation({ stepPerTick: 0.5, onTick: (v) => ticks.push(v) });
    vi.advanceTimersByTime(TICK_MS * 100);
    sim.stop();

    expect(Math.max(...ticks)).toBeLessThanOrEqual(1);
  });
});
