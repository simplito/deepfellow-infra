import * as React from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { Server, FileText } from "lucide-react";
import { NavUser } from "./nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from "@/components/ui/sidebar";

const navigationItems = [
  {
    title: "Services",
    url: "/dashboard",
    icon: Server,
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
                <span className="text-base font-semibold">DeepFellow Infra</span>
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
