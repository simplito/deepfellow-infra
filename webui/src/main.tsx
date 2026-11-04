/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import * as Sentry from "@sentry/react";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { StrictMode } from "react";
import ReactDOM from "react-dom/client";

import { RootErrorFallback } from "./components/RootErrorFallback.tsx";
import * as TanStackQueryProvider from "./integrations/tanstack-query/root-provider.tsx";
import { initSentry } from "./lib/sentry";

// Import the generated route tree
import { routeTree } from "./routeTree.gen";

import "./styles.css";
import reportWebVitals from "./reportWebVitals.ts";

const TanStackQueryProviderContext = TanStackQueryProvider.getContext();
const router = createRouter({
  routeTree,
  context: {
    ...TanStackQueryProviderContext,
  },
  defaultPreload: "intent",
  scrollRestoration: true,
  defaultStructuralSharing: true,
  defaultPreloadStaleTime: 0,
  // Used by every route without its own errorComponent (all of them), so a render error anywhere
  // shows a fallback instead of a blank page and is reported to Sentry.
  defaultErrorComponent: RootErrorFallback,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

function renderApp() {
  const rootElement = document.getElementById("app");
  if (rootElement && !rootElement.innerHTML) {
    // Errors the router's error boundaries catch are reported by RootErrorFallback; these
    // handlers cover the rest. Uncaught exceptions and unhandled rejections outside React are
    // reported by the SDK's global handlers.
    const root = ReactDOM.createRoot(rootElement, {
      onUncaughtError: Sentry.reactErrorHandler(),
      onRecoverableError: Sentry.reactErrorHandler(),
    });
    root.render(
      <StrictMode>
        <TanStackQueryProvider.Provider {...TanStackQueryProviderContext}>
          <RouterProvider router={router} />
        </TanStackQueryProvider.Provider>
      </StrictMode>,
    );
  }
}

// Sentry is set up before the first render (and before sign-in) so early crashes are reported too.
// initSentry never rejects and gives up after a short timeout, so it can't block startup.
initSentry().finally(renderApp);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals(console.log);
