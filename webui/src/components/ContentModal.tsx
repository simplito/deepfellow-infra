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
import { Loader2 } from "lucide-react";

interface ContentModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  content: string;
  wide?: boolean;
  pre?: boolean;
  isLoading?: boolean;
  onCancel?: () => void;
}

export function ContentModal({
  open,
  onOpenChange,
  title,
  content,
  wide = false,
  pre = true,
  isLoading = false,
  onCancel,
}: ContentModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={wide ? "sm:max-w-5xl" : "sm:max-w-4xl"}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-8 space-y-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-muted-foreground">Loading...</p>
          </div>
        ) : (
          <div className="max-h-[70vh] overflow-auto">
            {pre ? (
              <pre className="p-4 bg-muted rounded text-sm whitespace-pre-wrap">
                {content}
              </pre>
            ) : (
              <div className="p-4">{content}</div>
            )}
          </div>
        )}
        
        <div className="flex justify-end">
          {isLoading ? (
            <Button 
              onClick={onCancel || (() => onOpenChange(false))} 
              variant="outline"
            >
              Cancel
            </Button>
          ) : (
            <Button onClick={() => onOpenChange(false)}>OK</Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}


