/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

const TICK_MS = 50;
const MAX_SIMULATED = 0.962137;

export const COMPLETION_SMOOTH_MS = 5000;
export const COMPLETION_SMOOTH_MIN_MS = 300;

export interface SimulationConfig {
  stepPerTick: number;
  onTick: (value: number) => void;
  initialValue?: number;
}

export interface SimulationHandle {
  stop: () => void;
  smoothComplete: (durationMs?: number, onDone?: () => void, currentValue?: number) => void;
}

export function startProgressSimulation(config: SimulationConfig): SimulationHandle {
  let stopped = false;
  let simulated = Math.min(MAX_SIMULATED, Math.max(0, config.initialValue ?? 0));
  let smoothing = false;
  let smoothStep = 0;
  let onSmoothDone: (() => void) | undefined;

  const id = setInterval(() => {
    if (stopped) return;

    if (smoothing) {
      simulated = Math.min(1, simulated + smoothStep);
      config.onTick(simulated);
      if (simulated >= 1) {
        stopped = true;
        clearInterval(id);
        onSmoothDone?.();
      }
    } else {
      simulated += config.stepPerTick * (MAX_SIMULATED - simulated);
      config.onTick(simulated);
    }
  }, TICK_MS);

  return {
    stop: () => {
      stopped = true;
      clearInterval(id);
    },
    smoothComplete: (durationMs = COMPLETION_SMOOTH_MS, onDone?: () => void, currentValue?: number) => {
      // Sync with the store's current value — real SSE data may have advanced past the
      // simulation's internal position (whose updates were ignored by monotonicity rules),
      // so without this the animation would waste time replaying already-visible values.
      if (currentValue !== undefined && currentValue > simulated) simulated = currentValue;
      const remaining = 1 - simulated;
      if (remaining <= 0) {
        stopped = true;
        clearInterval(id);
        onDone?.();
        return;
      }
      onSmoothDone = onDone;
      const steps = Math.max(1, Math.round(durationMs / TICK_MS));
      smoothStep = remaining / steps;
      smoothing = true;
    },
  };
}

function parseSizeGb(sizeStr: string): number | null {
  const match = sizeStr.match(/([0-9.]+)\s*(MB|GB|TB)/i);
  if (!match) return null;

  const value = parseFloat(match[1]);
  const unit = match[2].toUpperCase();

  if (unit === "MB") return value / 1024;
  if (unit === "TB") return value * 1024;
  return value;
}

export function getStepPerTick(size: string | Record<string, string>, multiplier: number = 1): number {
  const candidates = typeof size === "string" ? [size] : Object.values(size);
  let maxGb: number | null = null;

  for (const s of candidates) {
    const gb = parseSizeGb(s);
    if (gb !== null && (maxGb === null || gb > maxGb)) {
      maxGb = gb;
    }
  }

  if (maxGb === null) return 0.0001;
  return multiplier * 0.001 / maxGb
}
