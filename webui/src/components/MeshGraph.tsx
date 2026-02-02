/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useMemo, useCallback, useState, useEffect } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  Handle,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type NodeTypes,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { MeshInfo, MeshInfoInfra } from "@/deepfellow/types";

interface MeshGraphProps {
  meshInfo: MeshInfo;
  currentServerName?: string;
  onNodeClick?: (node: Node) => void;
}

// Extended data types for nodes
export interface MainNodeData {
  label: string;
}

export interface ServiceNodeData {
  label: string;
  url: string;
  connection: MeshInfoInfra;
}

// Custom node component for main server (larger, distinct styling)
function MainNode({ data }: { data: { label: string } }) {
  return (
    <div className="px-6 py-3 bg-purple-600 text-white rounded-lg shadow-lg border-4 border-purple-400 min-w-[220px] text-center">
      <div className="font-bold text-base">{data.label}</div>
      <Handle type="source" position={Position.Bottom} style={{ background: "#8b5cf6" }} />
    </div>
  );
}

// Custom node component for services (rectangles)
function ServiceNode({ data }: { data: ServiceNodeData }) {
  return (
    <div className="px-4 py-2 bg-primary text-primary-foreground rounded-lg shadow-md border-2 border-primary/20 min-w-[200px] text-center cursor-pointer hover:border-primary/50 transition-colors">
      <Handle type="target" position={Position.Top} style={{ background: "#6366f1" }} />
      <div className="font-semibold text-sm">{data.label}</div>
      <div className="text-xs opacity-80 mt-1 truncate max-w-[180px]">{data.url}</div>
    </div>
  );
}

const nodeTypes: NodeTypes = {
  main: MainNode,
  service: ServiceNode,
};

export function MeshGraph({ meshInfo, currentServerName = "Current Server", onNodeClick }: MeshGraphProps) {
  const { nodes, edges } = useMemo(() => {
    const initialNodes: Node[] = [];
    const initialEdges: Edge[] = [];
    
    // Layout constants
    const mainNodeY = 50;
    const serviceY = mainNodeY + 150; // All services on same row below main
    const serviceXSpacing = 280;      // Horizontal spacing between services
    const graphCenterX = 500;         // Center of the graph horizontally
    
    // Create main/central node at top center (current server)
    const mainNodeId = "main-server";
    initialNodes.push({
      id: mainNodeId,
      type: "main",
      position: { x: graphCenterX, y: mainNodeY },
      data: {
        label: currentServerName,
      } satisfies MainNodeData,
    });

    // Calculate positions for services - spread horizontally in tree layout
    const totalConnections = meshInfo.connections?.length || 0;
    const totalWidth = totalConnections > 1 
      ? (totalConnections - 1) * serviceXSpacing 
      : 0;
    const startX = graphCenterX - totalWidth / 2;
    
    // Only create service nodes if connections exist
    if (meshInfo.connections && meshInfo.connections.length > 0) {
      meshInfo.connections.forEach((connection, connectionIndex) => {
        const serviceId = `service-${connectionIndex}`;
        const serviceX = startX + connectionIndex * serviceXSpacing;

        // Create service node with full connection data
        initialNodes.push({
          id: serviceId,
          type: "service",
          position: { x: serviceX, y: serviceY },
          data: {
            label: connection.name,
            url: connection.url,
            connection: connection,
          } satisfies ServiceNodeData,
        });

        // Create edge from main node to service
        initialEdges.push({
          id: `edge-main-${connectionIndex}`,
          source: mainNodeId,
          target: serviceId,
          type: "smoothstep",
          animated: false,
          style: { stroke: "#6366f1", strokeWidth: 2 },
        });
      });
    }

    return { nodes: initialNodes, edges: initialEdges };
  }, [meshInfo, currentServerName]);

  const [flowNodes, setFlowNodes] = useState<Node[]>(nodes);
  const [flowEdges, setFlowEdges] = useState<Edge[]>(edges);

  // Sync nodes and edges when they change from props
  useEffect(() => {
    setFlowNodes(nodes);
    setFlowEdges(edges);
  }, [nodes, edges]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      setFlowNodes((nds) => applyNodeChanges(changes, nds));
    },
    []
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setFlowEdges((eds) => applyEdgeChanges(changes, eds));
    },
    []
  );

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        onNodeClick(node);
      }
    },
    [onNodeClick]
  );


  return (
    <div className="w-full h-full min-h-[400px]">
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}

