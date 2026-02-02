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
import { SiteHeader } from "@/components/dashboard/site-header";

export const Route = createFileRoute("/dashboard/services/$serviceId")({
  component: ServiceModelsPage,
});

function ServiceModelsPage() {
  useRequireAuth();
  const { serviceId } = Route.useParams();

  return (
    <>
      <SiteHeader
        breadcrumbs={[
          { label: "Services", href: "/dashboard" },
          { label: serviceId },
        ]}
      />
      <div className="flex flex-1 flex-col">
        <div className="@container/main flex flex-1 flex-col gap-2">
          <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
            <ServiceModels serviceId={serviceId} />
          </div>
        </div>
      </div>
    </>
  );
}
