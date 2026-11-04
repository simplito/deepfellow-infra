/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

import type { ConfigEntry } from "@/deepfellow/types";

// A dynamic field's raw value is serialized with Python's str(bool), so this is how
// booleans are told apart from other editable fields (see design.md decision #2).
export function isBooleanValue(value: string): boolean {
  return value === "True" || value === "False";
}

// Observability settings are kept together, so a still-empty one (e.g. DF_SENTRY_ENVIRONMENT)
// isn't sorted away from the rest of its group.
const OBSERVABILITY_PREFIXES = ["DF_OTEL_", "DF_SENTRY_"];

function isObservabilityEntry(entry: ConfigEntry): boolean {
  return OBSERVABILITY_PREFIXES.some((prefix) => entry.key.startsWith(prefix));
}

function isFilled(entry: ConfigEntry): boolean {
  return entry.is_secret || Boolean(entry.value);
}

/**
 * Sort filled entries (secrets count as filled) before empty ones, keeping the server's order
 * otherwise. The observability group counts as filled when any of its entries is.
 */
export function sortConfigEntries(entries: ConfigEntry[]): ConfigEntry[] {
  const observabilityFilled = entries.some(
    (entry) => isObservabilityEntry(entry) && isFilled(entry),
  );
  const rank = (entry: ConfigEntry) =>
    (isObservabilityEntry(entry) ? observabilityFilled : isFilled(entry))
      ? 0
      : 1;
  return [...entries].sort((a, b) => rank(a) - rank(b));
}
