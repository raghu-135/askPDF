import React, { Component, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Chip, Typography, useTheme } from '@mui/material';
import ELK from 'elkjs/lib/elk.bundled.js';
import {
  Background,
  Edge,
  MarkerType,
  Node,
  OnNodesChange,
  OnNodeDrag,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import AgentGraphInspector from './AgentGraphInspector';
import AgentGraphNode from './AgentGraphNode';
import ReactFlowViewportChrome, { reactFlowChromeSx } from './ReactFlowViewportChrome';
import { applyTraceFocusToGraph, buildAgentGraph, getAgentGraphSpec } from './agent-graph-mapper';
import type { TraceRunView } from '../agent-debug/agent-trace-projection';
import type {
  AgentGraphEdge,
  AgentGraphMode,
  AgentGraphNode as AgentGraphNodeModel,
  AgentNodeCatalog,
  AgentGraphSelection,
  AgentNodeVisitRef,
  AgentTraceRefs,
} from './agent-graph-types';
import { applySelectedVisitOverlay } from './agent-node-visits';

const elk = new ELK();

const nodeTypes = {
  agentGraphNode: AgentGraphNode,
};

const AUTHORING_NODE_WIDTH = 230;
const AUTHORING_NODE_HEIGHT = 118;
const RUNTIME_NODE_WIDTH = 190;
const RUNTIME_NODE_HEIGHT = 86;

const getStatusEdgeColor = (edge: { selected?: boolean; active?: boolean }) => {
  if (edge.selected) return '#1976d2';
  if (edge.active) return '#2e7d32';
  return '#bdbdbd';
};

class AgentGraphErrorBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidUpdate(previousProps: { children: ReactNode; fallback: ReactNode }) {
    if (previousProps.children !== this.props.children && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

function AgentGraphFallback({ graph }: { graph: ReturnType<typeof buildAgentGraph> }) {
  if (graph.nodes.length === 0) {
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
        <Typography variant="caption" color="text.secondary">
          No agent graph topology is available for this run.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 1, borderRadius: 1, border: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
      <Typography variant="caption" sx={{ display: 'block', fontWeight: 700, mb: 0.75 }}>
        Agent graph
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
        {graph.nodes.map((node) => (
          <Chip
            key={node.id}
            size="small"
            color={node.status === 'error' ? 'error' : node.status === 'active' || node.status === 'completed' ? 'success' : node.status === 'planned' ? 'primary' : 'default'}
            variant={node.status === 'inactive' ? 'outlined' : 'filled'}
            label={`${node.label}: ${node.status}`}
          />
        ))}
      </Box>
    </Box>
  );
}

const layoutGraph = async (
  nodes: AgentGraphNodeModel[],
  edges: ReturnType<typeof buildAgentGraph>['edges'],
  direction: 'RIGHT' | 'DOWN',
) => {
  const runtimeLayout = direction === 'DOWN';
  const graph = {
    id: 'agent-run-graph',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': direction,
      'elk.spacing.nodeNode': direction === 'DOWN' ? '42' : '52',
      'elk.layered.spacing.nodeNodeBetweenLayers': direction === 'DOWN' ? '72' : '92',
      'elk.edgeRouting': 'ORTHOGONAL',
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: runtimeLayout ? RUNTIME_NODE_WIDTH : AUTHORING_NODE_WIDTH,
      height: runtimeLayout ? RUNTIME_NODE_HEIGHT : AUTHORING_NODE_HEIGHT,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };
  const result = await elk.layout(graph);
  const positions = new Map((result.children || []).map((node) => [node.id, { x: node.x || 0, y: node.y || 0 }]));
  return nodes.map((node): Node<AgentGraphNodeModel> => ({
    id: node.id,
    type: 'agentGraphNode',
    position: node.position || positions.get(node.id) || { x: 0, y: 0 },
    data: { ...node, layoutDirection: direction },
    selectable: true,
  }));
};

function AgentGraphCanvasInner({
  resolvedSpec,
  workflowId,
  traceView,
  focusedTraceRefs,
  nodeCatalog,
  mode,
  showInspector,
  onSelectionChange,
  onNodePositionChange,
  selectedVisitRef,
}: {
  resolvedSpec?: Record<string, any>;
  workflowId?: string;
  traceView?: TraceRunView;
  focusedTraceRefs?: AgentTraceRefs | null;
  nodeCatalog?: AgentNodeCatalog;
  mode: AgentGraphMode;
  showInspector: boolean;
  onSelectionChange?: (selection: AgentGraphSelection) => void;
  onNodePositionChange?: (nodeId: string, position: { x: number; y: number }) => void;
  selectedVisitRef?: AgentNodeVisitRef | null;
}) {
  const theme = useTheme();
  const { fitView } = useReactFlow();
  const [flowNodes, setFlowNodes] = useState<Node<AgentGraphNodeModel>[]>([]);
  const [selection, setSelection] = useState<AgentGraphSelection>(null);

  const graph = useMemo(() => {
    const graphSpec = getAgentGraphSpec(resolvedSpec, workflowId);
    let baseGraph;
    if (mode === 'run-debug' && traceView && graphSpec.nodes?.length) {
      baseGraph = buildAgentGraph(graphSpec, {
        route: traceView.route,
        metrics: traceView.metrics,
        nodeCatalog,
        nodeRows: traceView.operations.map((node) => ({
          ...node.raw,
          node: node.id,
          node_type: node.type,
          visit_index: node.visitIndex,
          status: node.status,
          skipped: node.skipped,
        })),
        toolRows: traceView.tools.map((tool) => ({
          ...tool.raw,
          tool_name: tool.name,
          caller_node: tool.callerNode,
          caller_node_type: tool.callerNodeType,
          caller_visit_index: tool.callerVisitIndex,
        })),
      });
    } else if (mode === 'run-debug' && traceView?.graph) {
      baseGraph = traceView.graph;
    } else {
      baseGraph = buildAgentGraph(graphSpec, { nodeCatalog });
    }
    const focusedGraph = applyTraceFocusToGraph(baseGraph, focusedTraceRefs);
    return traceView ? applySelectedVisitOverlay(focusedGraph, traceView.operations, selectedVisitRef) : focusedGraph;
  }, [focusedTraceRefs, mode, nodeCatalog, resolvedSpec, selectedVisitRef, workflowId, traceView]);
  const focusSignature = useMemo(() => JSON.stringify({
    node_ids: focusedTraceRefs?.node_ids || [],
    span_ids: focusedTraceRefs?.span_ids || [],
  }), [focusedTraceRefs]);
  const topologySignature = useMemo(() => JSON.stringify({
    nodes: graph.nodes.map((node) => ({ id: node.id, position: node.position })),
    edges: graph.edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target })),
  }), [graph.edges, graph.nodes]);
  const layoutDirection = mode === 'run-debug' ? 'DOWN' : 'RIGHT';
  const onSelectionChangeRef = useRef(onSelectionChange);

  useEffect(() => {
    onSelectionChangeRef.current = onSelectionChange;
  }, [onSelectionChange]);

  const flowEdges = useMemo((): Edge[] => graph.edges.map((edge: AgentGraphEdge) => {
    const color = getStatusEdgeColor(edge);
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      type: 'smoothstep',
      data: edge,
      selectable: true,
      animated: edge.selected,
      markerEnd: { type: MarkerType.ArrowClosed, color },
      style: {
        stroke: color,
        strokeWidth: edge.selected ? 3 : edge.active ? 2.25 : 1.5,
        opacity: edge.active || edge.selected ? 1 : 0.45,
      },
      labelStyle: {
        fill: edge.selected ? theme.palette.primary.main : theme.palette.text.secondary,
        fontSize: 13,
        fontWeight: edge.selected ? 700 : 500,
      },
      labelBgStyle: {
        fill: theme.palette.background.paper,
        fillOpacity: 0.85,
      },
    };
  }), [graph.edges, theme.palette.background.paper, theme.palette.primary.main, theme.palette.text.secondary]);

  useEffect(() => {
    let cancelled = false;
    layoutGraph(graph.nodes, graph.edges, layoutDirection)
      .then((layoutedNodes) => {
        if (cancelled) return;
        setFlowNodes(layoutedNodes);
        window.requestAnimationFrame(() => fitView({ padding: 0.08, duration: 200, maxZoom: 1 }));
      })
      .catch(() => {
        if (cancelled) return;
        setFlowNodes(graph.nodes.map((node, index) => ({
          id: node.id,
          type: 'agentGraphNode',
          position: layoutDirection === 'DOWN' ? { x: 0, y: index * 150 } : { x: index * 280, y: 0 },
          data: { ...node, layoutDirection },
          selectable: true,
        })));
      });
    return () => {
      cancelled = true;
    };
  }, [fitView, layoutDirection, topologySignature]);

  useEffect(() => {
    const graphNodesById = new Map(graph.nodes.map((node) => [node.id, node]));
    setFlowNodes((current) => current.map((node) => {
      const graphNode = graphNodesById.get(node.id);
      if (!graphNode) return node;
      return {
        ...node,
        selected: selectedVisitRef?.nodeId === node.id,
        data: { ...graphNode, layoutDirection },
      };
    }));
  }, [graph.nodes, layoutDirection, selectedVisitRef]);

  useEffect(() => {
    if (!selection) return;
    if (selection.kind === 'node' && !graph.nodes.some((node) => node.id === selection.node.id)) {
      setSelection(null);
    }
    if (selection.kind === 'edge' && !graph.edges.some((edge) => edge.id === selection.edge.id)) {
      setSelection(null);
    }
  }, [graph.edges, graph.nodes, selection]);

  useEffect(() => {
    const hasFocusRefs = Boolean(focusedTraceRefs?.node_ids?.length || focusedTraceRefs?.span_ids?.length);
    if (!hasFocusRefs) {
      setSelection(null);
      return;
    }
    const focusedNode = graph.nodes.find((node) => node.focused);
    if (focusedNode) {
      setSelection({ kind: 'node', node: focusedNode });
    }
  }, [focusedTraceRefs, focusSignature, graph.nodes]);

  useEffect(() => {
    if (!selectedVisitRef) return;
    const selectedNode = graph.nodes.find((node) => node.id === selectedVisitRef.nodeId);
    if (selectedNode) setSelection({ kind: 'node', node: selectedNode });
  }, [graph.nodes, selectedVisitRef]);

  if (graph.nodes.length === 0) {
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
        <Typography variant="caption" color="text.secondary">
          No agent graph topology is available for this run.
        </Typography>
      </Box>
    );
  }

  const canvasBg = theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.65)';
  const handleNodeDragStop: OnNodeDrag<Node<AgentGraphNodeModel>> = (_, node) => {
    onNodePositionChange?.(node.id, {
      x: Math.round(node.position.x),
      y: Math.round(node.position.y),
    });
  };
  const handleNodesChange: OnNodesChange<Node<AgentGraphNodeModel>> = (changes) => {
    setFlowNodes((current) => applyNodeChanges(changes, current));
  };
  const selectFromCanvas = (nextSelection: AgentGraphSelection) => {
    setSelection(nextSelection);
    onSelectionChangeRef.current?.(nextSelection);
  };

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 1, width: '100%' }}>
      <Box
        sx={{
          width: '100%',
          height: mode === 'run-debug' ? { xs: 460, md: 520 } : { xs: 420, md: 520 },
          minHeight: 320,
          maxHeight: '75vh',
          minWidth: 0,
          overflow: 'hidden',
          resize: mode === 'run-debug' ? 'vertical' : 'none',
          bgcolor: canvasBg,
          ...reactFlowChromeSx(theme, canvasBg),
        }}
      >
        <AgentGraphErrorBoundary fallback={<AgentGraphFallback graph={graph} />}>
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={nodeTypes}
            nodesConnectable={false}
            elementsSelectable
            onNodesChange={handleNodesChange}
            fitView
            fitViewOptions={{ padding: 0.08, maxZoom: 1 }}
            minZoom={0.55}
            maxZoom={1.8}
            zoomOnScroll={false}
            zoomOnPinch
            preventScrolling={false}
            onNodeClick={(_, node) => selectFromCanvas({ kind: 'node', node: node.data as AgentGraphNodeModel })}
            onNodeDragStop={handleNodeDragStop}
            onEdgeClick={(_, edge) => selectFromCanvas({ kind: 'edge', edge: edge.data as any })}
            onPaneClick={() => selectFromCanvas(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} size={1} />
            <ReactFlowViewportChrome showInteractive={false} miniMapPosition="top-left" />
          </ReactFlow>
        </AgentGraphErrorBoundary>
      </Box>
      {showInspector ? (
        <AgentGraphInspector selection={selection} />
      ) : null}
      {graph.executionPlan.length > 0 && (
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', gridColumn: '1 / -1' }}>
          Execution plan: {graph.executionPlan.join(' -> ')}
        </Typography>
      )}
    </Box>
  );
}

export default function AgentGraphCanvas(props: {
  resolvedSpec?: Record<string, any>;
  workflowId?: string;
  traceView?: TraceRunView;
  focusedTraceRefs?: AgentTraceRefs | null;
  nodeCatalog?: AgentNodeCatalog;
  mode?: AgentGraphMode;
  showInspector?: boolean;
  onSelectionChange?: (selection: AgentGraphSelection) => void;
  onNodePositionChange?: (nodeId: string, position: { x: number; y: number }) => void;
  selectedVisitRef?: AgentNodeVisitRef | null;
}) {
  return (
    <ReactFlowProvider>
      <AgentGraphCanvasInner
        {...props}
        mode={props.mode || 'run-debug'}
        showInspector={props.showInspector !== false}
      />
    </ReactFlowProvider>
  );
}
