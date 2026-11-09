/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { Service } from "@/deepfellow/types";

export const LOAD_ERROR_HINT =
  "The services configuration could not be read. Fix the file and restart DeepFellow Infra.";

export const CLOUD_DISABLED_HINT =
  "Cloud services are disabled. Enable cloud to install this service.";

export function getActionBlockReason(
  service: Pick<Service, "is_cloud" | "disabled_reason">,
  loadError: string | null | undefined,
  cloudEnabled: boolean | undefined,
): string | null {
  if (loadError) return LOAD_ERROR_HINT;
  if (service.is_cloud && cloudEnabled === false) return CLOUD_DISABLED_HINT;
  return service.disabled_reason || null;
}
