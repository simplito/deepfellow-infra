/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import * as Sentry from "@sentry/react";
import type { ErrorComponentProps } from "@tanstack/react-router";
import { useEffect } from "react";

// The router's default error component: rendered in place of a failed route, up to the whole app,
// so it must not depend on providers set up by the root component (modals, toaster).
export function RootErrorFallback({ error, reset }: ErrorComponentProps) {
  useEffect(() => {
    // A no-op while Sentry is off.
    Sentry.captureException(error);
  }, [error]);

  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Something went wrong</CardTitle>
          <CardDescription>
            An unexpected error occurred. Try again, or reload the page if the
            problem persists.
          </CardDescription>
        </CardHeader>
        {error instanceof Error && error.message && (
          <CardContent>
            <p className="font-mono text-sm text-muted-foreground break-all">
              {error.message}
            </p>
          </CardContent>
        )}
        <CardFooter className="gap-2">
          <Button onClick={reset}>Try again</Button>
          <Button variant="outline" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
