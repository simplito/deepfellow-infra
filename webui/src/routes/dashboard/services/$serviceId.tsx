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
