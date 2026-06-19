/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * Propose an endpoint prefix from a model id, mirroring the backend
 * `normalize_name` (server/utils/core.py): lowercase, keep ASCII alphanumerics,
 * replace every other character with `_`.
 */
export function proposePrefix(modelId: string): string {
  return Array.from(modelId.toLowerCase(), (ch) =>
    /[a-z0-9]/.test(ch) ? ch : "_",
  ).join("");
}
