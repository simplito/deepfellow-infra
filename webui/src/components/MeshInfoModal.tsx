/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { MeshGraph, type ServiceNodeData } from "./MeshGraph";
import type { ModalProps } from "@/hooks/use-modal";
import type { Node } from "@xyflow/react";
import type { MeshInfo, MeshInfoInfra } from "@/deepfellow/types";
import { Server, Box } from "lucide-react";

interface MeshInfoModalProps extends ModalProps {
  meshInfo: MeshInfo;
}

// Type badge color mapping
function getModelTypeColor(type: string): string {
  switch (type) {
    case "llm":
      return "bg-blue-500 hover:bg-blue-600";
    case "txt2img":
      return "bg-green-500 hover:bg-green-600";
    case "custom":
      return "bg-orange-500 hover:bg-orange-600";
    case "tts":
      return "bg-pink-500 hover:bg-pink-600";
    case "stt":
      return "bg-cyan-500 hover:bg-cyan-600";
    case "embedding":
      return "bg-yellow-500 hover:bg-yellow-600";
    case "lora":
      return "bg-purple-500 hover:bg-purple-600";
    default:
      return "bg-gray-500 hover:bg-gray-600";
  }
}

export function MeshInfoModal({
  open,
  onOpenChange,
  meshInfo,
}: MeshInfoModalProps) {
  const [selectedServer, setSelectedServer] = useState<MeshInfoInfra | null>(null);

  const handleNodeClick = (node: Node) => {
    if (node.type === "service") {
      const nodeData = node.data as unknown as ServiceNodeData;
      setSelectedServer(nodeData.connection);
    } else if (node.type === "main") {
      // Clear selection when clicking main server
      setSelectedServer(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-6xl h-[700px] flex flex-col">
        <DialogHeader>
          <DialogTitle>Mesh Information</DialogTitle>
        </DialogHeader>
        
        <div className="flex-1 min-h-0 mt-4">
          <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
            <ResizablePanel defaultSize={70} minSize={50}>
              <MeshGraph meshInfo={meshInfo} onNodeClick={handleNodeClick} />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={30} minSize={20}>
              <ScrollArea className="h-full">
                <div className="p-4">
                  {selectedServer ? (
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Server className="h-5 w-5 text-primary" />
                          <h3 className="font-semibold text-lg">{selectedServer.name}</h3>
                        </div>
                        <p className="text-sm text-muted-foreground break-all">
                          {selectedServer.url}
                        </p>
                      </div>
                      
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <Box className="h-4 w-4 text-muted-foreground" />
                          <h4 className="font-medium text-sm text-muted-foreground">
                            Models ({selectedServer.models.length})
                          </h4>
                        </div>
                        <div className="space-y-2">
                          {selectedServer.models.map((model) => (
                            <div
                              key={model.name}
                              className="flex items-center justify-between gap-2 p-2 rounded-md bg-muted/50"
                            >
                              <span className="text-sm font-medium truncate">
                                {model.name}
                              </span>
                              <Badge className={`${getModelTypeColor(model.type)} text-white text-xs`}>
                                {model.type}
                              </Badge>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-[300px] text-center text-muted-foreground">
                      <Server className="h-12 w-12 mb-4 opacity-50" />
                      <p className="text-sm">
                        Click on a server node in the graph to view its details and models.
                      </p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
        
        <div className="flex justify-end mt-4">
          <Button onClick={() => onOpenChange(false)}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

