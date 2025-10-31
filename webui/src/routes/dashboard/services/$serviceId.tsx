/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { createFileRoute } from "@tanstack/react-router";
import { ServiceModels } from "@/components/ServiceModels";
import { useRequireAuth } from "@/hooks/use-auth";

export const Route = createFileRoute("/dashboard/services/$serviceId")({
  component: ServiceModelsPage,
});

function ServiceModelsPage() {
  useRequireAuth();
  const { serviceId } = Route.useParams();

  return <ServiceModels serviceId={serviceId} />;
}
