import { Outlet, createFileRoute, useNavigate } from "@tanstack/react-router";
import { AppSidebar } from "@/components/dashboard/app-sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { AdminApiKeyStorage } from "@/deepfellow/client";

export const Route = createFileRoute("/dashboard")({
  component: RouteComponent,
});

function RouteComponent() {
  const navigate = useNavigate();

  const handleLogout = () => {
    AdminApiKeyStorage.clear();
    navigate({ to: "/sign-in" });
  };

  return (
    <SidebarProvider>
      <AppSidebar variant="inset" onLogout={handleLogout} />
      <SidebarInset>
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  );
}
