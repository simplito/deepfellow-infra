/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import {
  Background,
  Controls,
  type Edge,
  Handle,
  type Node,
  type NodeTypes,
  type OnEdgesChange,
  type OnNodesChange,
  Position,
  ReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";
import type { MeshTopologyNode } from "@/deepfellow/types";

interface MeshGraphProps {
  meshInfo: MeshTopologyNode[];
  onNodeClick?: (node: Node) => void;
}

export interface MainNodeData {
  label: string;
  hasParent: boolean;
  topologyNode: MeshTopologyNode;
}

export interface ServiceNodeData {
  label: string;
  url: string;
  topologyNode: MeshTopologyNode;
}

function MainNode({ data }: { data: MainNodeData }) {
  return (
    <div className="px-6 py-3 bg-purple-600 text-white rounded-lg shadow-lg border-4 border-purple-400 min-w-[220px] text-center">
      {data.hasParent && (
        <Handle
          type="target"
          position={Position.Top}
          style={{ background: "#8b5cf6" }}
        />
      )}
      <div className="font-bold text-base">{data.label}</div>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#8b5cf6" }}
      />
    </div>
  );
}

function ServiceNode({ data }: { data: ServiceNodeData }) {
  return (
    <div className="px-4 py-2 bg-primary text-primary-foreground rounded-lg shadow-md border-2 border-primary/20 min-w-[200px] text-center cursor-pointer hover:border-primary/50 transition-colors">
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#6366f1" }}
      />
      <div className="font-semibold text-sm">{data.label}</div>
      <div className="text-xs opacity-80 mt-1 truncate max-w-[180px]">
        {data.url}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#6366f1" }}
      />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  main: MainNode,
  service: ServiceNode,
};

const DEPTH_Y_SPACING = 170;
const MIN_X_SPACING = 280;

function buildTree(
  nodes: MeshTopologyNode[],
  parentId: string | null,
  depth: number,
  centerX: number,
  initialNodes: Node[],
  initialEdges: Edge[],
  nodeCounter: { value: number },
): void {
  if (nodes.length === 0) return;

  const totalWidth = Math.max((nodes.length - 1) * MIN_X_SPACING, 0);
  const startX = centerX - totalWidth / 2;
  const y = 50 + depth * DEPTH_Y_SPACING;

  nodes.forEach((node, i) => {
    const id = node.you_are_here
      ? "main-server"
      : `service-${nodeCounter.value++}`;
    const x = startX + i * MIN_X_SPACING;

    if (node.you_are_here) {
      initialNodes.push({
        id,
        type: "main",
        position: { x, y },
        data: {
          label: node.name,
          hasParent: parentId !== null,
          topologyNode: node,
        } satisfies MainNodeData,
      });
    } else {
      initialNodes.push({
        id,
        type: "service",
        position: { x, y },
        data: {
          label: node.name,
          url: node.url,
          topologyNode: node,
        } satisfies ServiceNodeData,
      });
    }

    if (parentId !== null) {
      initialEdges.push({
        id: `edge-${parentId}-${id}`,
        source: parentId,
        target: id,
        type: "smoothstep",
        animated: false,
        style: { stroke: "#6366f1", strokeWidth: 2 },
      });
    }

    buildTree(
      node.children,
      id,
      depth + 1,
      x,
      initialNodes,
      initialEdges,
      nodeCounter,
    );
  });
}

export function MeshGraph({ meshInfo, onNodeClick }: MeshGraphProps) {
  const { nodes, edges } = useMemo(() => {
    const initialNodes: Node[] = [];
    const initialEdges: Edge[] = [];

    buildTree(meshInfo, null, 0, 500, initialNodes, initialEdges, { value: 0 });

    return { nodes: initialNodes, edges: initialEdges };
  }, [meshInfo]);

  const [flowNodes, setFlowNodes] = useState<Node[]>(nodes);
  const [flowEdges, setFlowEdges] = useState<Edge[]>(edges);

  useEffect(() => {
    setFlowNodes(nodes);
    setFlowEdges(edges);
  }, [nodes, edges]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setFlowNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setFlowEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (onNodeClick) onNodeClick(node);
    },
    [onNodeClick],
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
