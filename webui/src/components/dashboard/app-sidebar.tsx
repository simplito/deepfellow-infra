import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { apiClient } from "@/deepfellow/client";
import { useQuery } from "@tanstack/react-query";
import { Link, useRouterState } from "@tanstack/react-router";
import { FileText, Server, Settings, TriangleAlert } from "lucide-react";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type * as React from "react";
import { NavUser } from "./nav-user";

const navigationItems = [
  {
    title: "Services",
    url: "/dashboard",
    icon: Server,
  },
  {
    title: "Configuration",
    url: "/dashboard/config",
    icon: Settings,
  },
  {
    title: "Warnings",
    url: "/dashboard/warnings",
    icon: TriangleAlert,
  },
  {
    title: "Documentation",
    url: "/docs",
    icon: FileText,
    external: true,
  },
];

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  onLogout: () => void;
}

export function AppSidebar({ onLogout, ...props }: AppSidebarProps) {
  const routerState = useRouterState();
  const currentPath = routerState.location.pathname;

  const { data: warnings } = useQuery({
    queryKey: ["admin", "warnings"],
    queryFn: () => apiClient.listAdminWarnings(),
    refetchInterval: 30000,
  });
  const warningsCount =
    (warnings?.list.length ?? 0) + (warnings?.load_error ? 1 : 0);

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <Link to="/dashboard">
                <Server className="size-5" />
                <span className="text-base font-semibold">
                  DeepFellow Infra
                </span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => {
                const isActive = !item.external && currentPath === item.url;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild isActive={isActive}>
                      {item.external ? (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <item.icon />
                          <span>{item.title}</span>
                        </a>
                      ) : (
                        <Link to={item.url}>
                          <item.icon />
                          <span>{item.title}</span>
                        </Link>
                      )}
                    </SidebarMenuButton>
                    {item.url === "/dashboard/warnings" &&
                      warningsCount > 0 && (
                        <SidebarMenuBadge>{warningsCount}</SidebarMenuBadge>
                      )}
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <NavUser user={{ name: "Admin" }} onLogout={onLogout} />
      </SidebarFooter>
    </Sidebar>
  );
}
