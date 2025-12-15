/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import {
  Outlet,
  createRootRouteWithContext,
  useRouter,
} from "@tanstack/react-router";
import { useEffect } from "react";

import { useAuth } from "../hooks/use-auth";
import { Toaster } from "../components/ui/sonner";
import { ModalProvider } from "../hooks/use-modal";

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
    <ModalProvider>
      <Outlet />
      <Toaster />
    </ModalProvider>
  );
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
  component: RootComponent,
});
