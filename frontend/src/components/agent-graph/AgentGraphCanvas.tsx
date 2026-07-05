import React, { useEffect, useMemo, useState } from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import ELK from 'elkjs/lib/elk.bundled.js';
import {
  Background,
  Controls,
  Edge,
  MarkerType,
  Node,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import AgentGraphInspector from './AgentGraphInspector';
import AgentGraphNode from './AgentGraphNode';
import { buildAgentGraph, getAgentGraphSpec } from './agent-graph-mapper';
import type { TraceRunView } from '../agent-debug/agent-trace-projection';
import type {
  AgentGraphEdge,
  AgentGraphMode,
  AgentGraphNode as AgentGraphNodeModel,
  AgentGraphSelection,
} from './agent-graph-types';

const elk = new ELK();

const nodeTypes = {
  agentGraphNode: AgentGraphNode,
};

const NODE_WIDTH = 230;
const NODE_HEIGHT = 118;

const getStatusEdgeColor = (edge: { selected?: boolean; active?: boolean }) => {
  if (edge.selected) return '#1976d2';
  if (edge.active) return '#2e7d32';
  return '#bdbdbd';
};

const layoutGraph = async (
  nodes: AgentGraphNodeModel[],
  edges: ReturnType<typeof buildAgentGraph>['edges'],
  direction: 'RIGHT' | 'DOWN',
) => {
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
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
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
    position: positions.get(node.id) || { x: 0, y: 0 },
    data: { ...node, layoutDirection: direction },
    draggable: false,
    selectable: true,
  }));
};

function AgentGraphCanvasInner({
  resolvedSpec,
  templateId,
  traceView,
  mode,
}: {
  resolvedSpec?: Record<string, any>;
  templateId?: string;
  traceView?: TraceRunView;
  mode: AgentGraphMode;
}) {
  const theme = useTheme();
  const { fitView } = useReactFlow();
  const [flowNodes, setFlowNodes] = useState<Node<AgentGraphNodeModel>[]>([]);
  const [selection, setSelection] = useState<AgentGraphSelection>(null);

  const graph = useMemo(() => {
    if (mode === 'run-debug' && traceView?.graph) {
      return traceView.graph;
    }
    const graphSpec = getAgentGraphSpec(resolvedSpec, templateId);
    return buildAgentGraph(graphSpec);
  }, [mode, resolvedSpec, templateId, traceView]);
  const layoutDirection = mode === 'run-debug' ? 'DOWN' : 'RIGHT';

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
          draggable: false,
          selectable: true,
        })));
      });
    return () => {
      cancelled = true;
    };
  }, [fitView, graph.edges, graph.nodes, layoutDirection]);

  useEffect(() => {
    if (!selection) return;
    if (selection.kind === 'node' && !graph.nodes.some((node) => node.id === selection.node.id)) {
      setSelection(null);
    }
    if (selection.kind === 'edge' && !graph.edges.some((edge) => edge.id === selection.edge.id)) {
      setSelection(null);
    }
  }, [graph.edges, graph.nodes, selection]);

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

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 1, width: '100%' }}>
      <Box
        sx={{
          width: '100%',
          height: mode === 'run-debug' ? { xs: 620, md: 680 } : { xs: 420, md: 520 },
          minHeight: 360,
          maxHeight: '85vh',
          minWidth: 0,
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          overflow: 'hidden',
          resize: mode === 'run-debug' ? 'vertical' : 'none',
          bgcolor: canvasBg,
          '& .react-flow__controls': {
            overflow: 'hidden',
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            boxShadow: theme.shadows[2],
            bgcolor: canvasBg,
          },
          '& .react-flow__controls-button': {
            width: 30,
            height: 30,
            bgcolor: canvasBg,
            color: 'text.primary',
            borderBottom: 1,
            borderBottomColor: 'divider',
          },
          '& .react-flow__controls-button:hover': {
            bgcolor: 'action.hover',
          },
          '& .react-flow__controls-button svg': {
            fill: 'currentColor',
          },
        }}
      >
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          nodesDraggable={mode === 'builder'}
          nodesConnectable={mode === 'builder'}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.08, maxZoom: 1 }}
          minZoom={0.55}
          maxZoom={1.8}
          zoomOnScroll={false}
          zoomOnPinch
          preventScrolling={false}
          onNodeClick={(_, node) => setSelection({ kind: 'node', node: node.data as AgentGraphNodeModel })}
          onEdgeClick={(_, edge) => setSelection({ kind: 'edge', edge: edge.data as any })}
          onPaneClick={() => setSelection(null)}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </Box>
      <AgentGraphInspector selection={selection} />
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
  templateId?: string;
  traceView?: TraceRunView;
  mode?: AgentGraphMode;
}) {
  return (
    <ReactFlowProvider>
      <AgentGraphCanvasInner {...props} mode={props.mode || 'run-debug'} />
    </ReactFlowProvider>
  );
}
