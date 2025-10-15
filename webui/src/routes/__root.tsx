import {
  Outlet,
  createRootRouteWithContext,
  useRouter,
} from "@tanstack/react-router";
import { useEffect } from "react";

import { useAuth } from "../hooks/use-auth";
import { Toaster } from "../components/ui/sonner";

import type { QueryClient } from "@tanstack/react-query";

interface MyRouterContext {
  queryClient: QueryClient;
}

function RootComponent() {
  const { apiKey, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !apiKey && router.state.location.pathname === "/") {
      router.navigate({ to: "/sign-in" });
    } else if (!isLoading && apiKey && router.state.location.pathname === "/") {
      router.navigate({ to: "/dashboard" });
    }
  }, [apiKey, isLoading, router]);

  return (
    <>
      <Outlet />
      <Toaster />
    </>
  );
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
  component: RootComponent,
});
