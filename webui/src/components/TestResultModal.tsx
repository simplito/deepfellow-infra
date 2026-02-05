/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";
import type { TestResult } from "@/deepfellow/types";

interface TestResultModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  result: TestResult;
  isLoading?: boolean;
  onCancel?: () => void;
}

export function TestResultModal({ open, onOpenChange, result, isLoading = false, onCancel }: TestResultModalProps) {
  const renderOutput = () => {
    if (!result.output) return null;

    if (typeof result.output === "string") {
      return <div className="mt-4 p-4 bg-muted rounded">{result.output}</div>;
    }

    const { content_type, data } = result.output;

    if (content_type.startsWith("audio/")) {
      return (
        <div className="mt-4">
          <audio controls className="w-full">
            <source src={`data:${content_type};base64,${data}`} type={content_type} />
          </audio>
        </div>
      );
    }

    if (content_type.startsWith("image/")) {
      return (
        <div className="mt-4">
          <img
            src={`data:${content_type};base64,${data}`}
            alt="Test output"
            className="max-w-full max-h-64 rounded"
          />
        </div>
      );
    }

    return (
      <div className="mt-4 p-4 bg-muted rounded">
        Content with type {content_type}
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-6xl">
        <DialogHeader>
          <DialogTitle>Test Result</DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-8 space-y-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-muted-foreground">Running test...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {result.error && (
              <div className="flex items-center gap-2 text-destructive">
                <XCircle className="h-5 w-5" />
                <span className="font-semibold">Test failed!</span>
              </div>
            )}

            {result.result && (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle2 className="h-5 w-5" />
                <span className="font-semibold">Test passed!</span>
              </div>
            )}

            {result.output && (
              <div>
                <h3 className="font-semibold mb-2">Output:</h3>
                {renderOutput()}
              </div>
            )}

            {result.details && (
              <div>
                <h3 className="font-semibold mb-2">Details:</h3>
                <pre className="p-4 bg-muted rounded text-sm overflow-auto max-h-48 whitespace-pre-wrap" style={({lineBreak: "anywhere"})}>
                  {JSON.stringify(result.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        <div className="flex justify-end mt-4">
          {isLoading ? (
            <Button 
              onClick={onCancel || (() => onOpenChange(false))} 
              variant="outline"
            >
              Cancel
            </Button>
          ) : (
            <Button onClick={() => onOpenChange(false)}>
              OK
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}


