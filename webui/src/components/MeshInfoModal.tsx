import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { apiClient } from "@/deepfellow/client";
import type { MeshTopologyNode } from "@/deepfellow/types";
import type { ModalProps } from "@/hooks/use-modal";
import { useQuery } from "@tanstack/react-query";
import type { Node } from "@xyflow/react";
import { AlertCircle, Loader2, Server } from "lucide-react";
import { MeshModelGroups } from "./MeshModelGroups";
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
  type MainNodeData,
  MeshGraph,
  type ServiceNodeData,
} from "./MeshGraph";

interface MeshInfoModalProps extends ModalProps {}

export function MeshInfoModal({ open, onOpenChange }: MeshInfoModalProps) {
  const [selectedNode, setSelectedNode] = useState<MeshTopologyNode | null>(
    null,
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["meshTopology"],
    queryFn: () => apiClient.getMeshTopology(),
    refetchInterval: 3000,
    enabled: open,
  });

  const handleNodeClick = (node: Node) => {
    if (node.type === "service") {
      const nodeData = node.data as unknown as ServiceNodeData;
      setSelectedNode(nodeData.topologyNode);
    } else if (node.type === "main") {
      const nodeData = node.data as unknown as MainNodeData;
      setSelectedNode(nodeData.topologyNode);
    }
  };

  const renderGraph = () => {
    if (isLoading && !data) {
      return (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      );
    }
    if (isError && !data) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
          <AlertCircle className="h-8 w-8" />
          <p className="text-sm">Failed to load topology</p>
        </div>
      );
    }
    return <MeshGraph meshInfo={data ?? []} onNodeClick={handleNodeClick} />;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-6xl h-[700px] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Mesh Information
            {isError && data && (
              <AlertCircle className="h-4 w-4 text-amber-500" />
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 min-h-0 mt-4">
          <ResizablePanelGroup
            direction="horizontal"
            className="h-full rounded-lg border"
          >
            <ResizablePanel defaultSize={70} minSize={50}>
              {renderGraph()}
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={30} minSize={20}>
              <ScrollArea className="h-full">
                <div className="p-4">
                  {selectedNode ? (
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Server className="h-5 w-5 text-primary" />
                          <h3 className="font-semibold text-lg">
                            {selectedNode.name}
                          </h3>
                          {selectedNode.you_are_here && (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-purple-600 text-white font-medium">
                              You are here
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground break-all">
                          {selectedNode.url}
                        </p>
                      </div>
                      {selectedNode.models.length > 0 && (
                        <div>
                          <p className="text-sm text-muted-foreground font-medium mb-2">
                            Models ({selectedNode.models.length})
                          </p>
                          <MeshModelGroups models={selectedNode.models} />
                        </div>
                      )}
                      {selectedNode.children.length > 0 && (
                        <div>
                          <p className="text-sm text-muted-foreground font-medium mb-2">
                            Sub-connections ({selectedNode.children.length})
                          </p>
                          <div className="space-y-1">
                            {selectedNode.children.map((child) => (
                              <div
                                key={child.url}
                                className="text-sm px-2 py-1 rounded-md bg-muted/50 truncate"
                              >
                                {child.name}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-[300px] text-center text-muted-foreground">
                      <Server className="h-12 w-12 mb-4 opacity-50" />
                      <p className="text-sm">
                        Click on a server node in the graph to view its details.
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
