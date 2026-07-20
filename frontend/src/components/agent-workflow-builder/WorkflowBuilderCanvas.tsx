import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import {
  Background,
  Connection,
  Controls,
  Edge,
  FinalConnectionState,
  MarkerType,
  MiniMap,
  Node,
  OnConnectStartParams,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
} from '@xyflow/react';
import ELK from 'elkjs/lib/elk.bundled.js';
import { Alert, Button } from '@mui/material';
import '@xyflow/react/dist/style.css';

import type { AgentWorkflowCatalogResponse } from '../../lib/api';
import type { AgentGraphNode as AgentGraphNodeModel } from '../agent-graph/agent-graph-types';
import AgentGraphNode from '../agent-graph/AgentGraphNode';
import {
  canConnectNodes,
  getAllowedRouteFunctionsForNode,
  getRouteLabelsForFunction,
  type AgentWorkflowBuilderState,
} from '../../lib/agent-workflow-builder';
import type { BuilderSelection, BuilderValidationIssue } from './types';

const elk = new ELK();
const nodeTypes = { agentGraphNode: AgentGraphNode };
const WIDTH = 230;
const HEIGHT = 150;

const layout = async (nodes: Node<AgentGraphNodeModel>[], edges: Edge[]) => {
  const graphNodes = nodes.filter((node) => node.data.type !== 'canvas_note');
  const decorationNodes = nodes.filter((node) => node.data.type === 'canvas_note');
  const result = await elk.layout({
    id: 'builder',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': '72',
      'elk.layered.spacing.nodeNodeBetweenLayers': '132',
      'elk.edgeRouting': 'ORTHOGONAL',
    },
    children: graphNodes.map((node) => ({
      id: node.id,
      width: node.measured?.width || WIDTH,
      height: Math.max(
        node.measured?.height || 0,
        node.id === 'START' || node.id === 'END'
          ? 112
          : HEIGHT + Math.max(0, (node.data.outputPorts?.length || 1) - 1) * 24,
      ),
    })),
    edges: edges
      .filter((edge) => graphNodes.some((node) => node.id === edge.source) && graphNodes.some((node) => node.id === edge.target))
      .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
  });
  const positions = new Map((result.children || []).map((node) => [node.id, { x: node.x || 0, y: node.y || 0 }]));
  const graphBottom = Math.max(0, ...(result.children || []).map((node) => (node.y || 0) + (node.height || HEIGHT)));
  decorationNodes.forEach((node, index) => {
    positions.set(node.id, {
      x: (index % 3) * (WIDTH + 32),
      y: graphBottom + 96 + Math.floor(index / 3) * (HEIGHT + 32),
    });
  });
  return nodes.map((node) => ({ ...node, position: positions.get(node.id) || node.position }));
};

function Canvas({
  catalog,
  state,
  selection,
  issues,
  disabled,
  onSelectionChange,
  onConnectNodes,
  onRemoveEdge,
  onRemoveNode,
  onNodePositionChange,
  onAddNodeAt,
  onRequestAddPrevious,
  onRequestAddNext,
  onUpdateNote,
  onRemoveNote,
  onPositionsChange,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  selection: BuilderSelection;
  issues: BuilderValidationIssue[];
  disabled?: boolean;
  onSelectionChange: (selection: BuilderSelection) => void;
  onConnectNodes: (source: string, target: string, route?: string) => void;
  onRemoveEdge: (index: number, route?: string) => void;
  onRemoveNode: (id: string) => void;
  onNodePositionChange: (id: string, position: { x: number; y: number }) => void;
  onAddNodeAt: (nodeType: string, position: { x: number; y: number }) => void;
  onRequestAddPrevious: (target: string) => void;
  onRequestAddNext: (source: string, route?: string) => void;
  onUpdateNote: (id: string, position: { x: number; y: number }) => void;
  onRemoveNote: (id: string) => void;
  onPositionsChange: (positions: Record<string, { x: number; y: number }>) => void;
}) {
  const [flowNodes, setFlowNodes] = useState<Node<AgentGraphNodeModel>[]>([]);
  const didInitialLayout = useRef(false);
  const [connecting, setConnecting] = useState<{ source: string; route?: string } | null>(null);
  const [connectionMessage, setConnectionMessage] = useState<string | null>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();
  useEffect(() => {
    if (!connecting) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setConnecting(null);
      setConnectionMessage(null);
    };
    window.addEventListener('keydown', cancel);
    return () => window.removeEventListener('keydown', cancel);
  }, [connecting]);

  const graphNodes = useMemo(() => {
    const virtual = [
      { id: 'START', type: 'START', position: { x: 0, y: 0 }, label: 'Start', category: 'context' },
      ...state.nodes.map((node) => {
        const entry = catalog.node_catalog[node.type];
        const routeFn = getAllowedRouteFunctionsForNode(catalog, node.type)[0];
        const labels = routeFn ? getRouteLabelsForFunction(catalog, routeFn) : [];
        const routeOptions = routeFn ? catalog.route_functions[routeFn]?.route_options || {} : {};
        const outputPorts = labels === null
          ? (node.hitl?.allowed_actions || []).map((id) => ({ id, label: id.replace(/_/g, ' ') }))
          : labels?.map((id) => ({
            id,
            label: routeOptions[id]?.display_name || id.replace(/_/g, ' '),
            description: routeOptions[id]?.description,
          }));
        return {
          id: node.id,
          type: node.type,
          position: node.position || { x: 0, y: 0 },
          label: node.label || entry?.display_name || node.type,
          category: entry?.category,
          description: node.description || entry?.ui?.summary,
          outputPorts: outputPorts?.length ? outputPorts : [{ id: 'default', label: entry?.ui?.output_label || 'Next' }],
          usesLlm: entry?.ui?.uses_llm,
          usesTools: entry?.ui?.uses_tools,
        };
      }),
      ...(state.builder_ui?.notes || []).map((note) => ({
        id: note.id,
        type: 'canvas_note',
        position: note.position,
        label: 'Note',
        category: 'note',
        description: note.text,
        isNote: true,
      })),
      ...(state.builder_ui?.groups || []).map((group) => ({
        id: group.id,
        type: 'canvas_note',
        position: group.position || { x: 40, y: 40 },
        label: group.label,
        category: 'note',
        description: `${group.node_ids.length} grouped steps: ${group.node_ids.join(', ')}`,
        isNote: true,
      })),
      { id: 'END', type: 'END', position: { x: 0, y: 0 }, label: 'End', category: 'answer' },
    ];
    return virtual.map((item): Node<AgentGraphNodeModel> => {
      const isSentinel = item.id === 'START' || item.id === 'END';
      const isNote = 'isNote' in item && item.isNote;
      const compatible = !connecting || item.id === connecting.source
        ? undefined
        : canConnectNodes(catalog, state, connecting.source, item.id).ok;
      const compatibilityReason = !connecting || compatible
        ? undefined
        : canConnectNodes(catalog, state, connecting.source, item.id).reason;
      return {
        id: item.id,
        type: 'agentGraphNode',
        position: item.position,
        deletable: !isSentinel,
        data: {
          id: item.id,
          type: item.type,
          label: item.label,
          instanceLabel: isSentinel ? item.label : item.id,
          category: item.category,
          description: 'description' in item ? item.description : isSentinel ? (item.id === 'START' ? 'The workflow begins here.' : 'The workflow finishes here.') : undefined,
          position: item.position,
          status: 'inactive',
          toolSummaries: [],
          warningCount: 0,
          errorCount: 0,
          sourceCount: 0,
          artifactCount: 0,
          rawEvents: [],
          authoring: !disabled && !isNote,
          outputPorts: item.id === 'END' ? [] : 'outputPorts' in item ? item.outputPorts : [{ id: 'default', label: 'Begin' }],
          compatible,
          compatibilityReason,
          issueCount: issues.filter((issue) => issue.selection?.kind === 'node' && issue.selection.nodeId === item.id).length,
          usesLlm: 'usesLlm' in item ? item.usesLlm : false,
          usesTools: 'usesTools' in item ? item.usesTools : false,
          onAddPrevious: isNote ? undefined : onRequestAddPrevious,
          onAddNext: isNote ? undefined : onRequestAddNext,
        },
        selected: selection?.kind === 'node' && selection.nodeId === item.id,
      };
    });
  }, [catalog, connecting, disabled, issues, onRequestAddNext, onRequestAddPrevious, selection, state]);

  const graphEdges = useMemo((): Edge[] => state.edges.flatMap((edge, edgeIndex) => {
    if (edge.conditional) {
      return Object.entries(edge.routes || {}).map(([route, target]) => ({
        id: `${edgeIndex}:${edge.from}:${route}:${target}`,
        source: edge.from,
        sourceHandle: route,
        target,
        label: catalog.route_functions[edge.route_fn || '']?.route_options?.[route]?.display_name || route.replace(/_/g, ' '),
        markerEnd: { type: MarkerType.ArrowClosed },
        data: { edgeIndex, route },
        type: 'smoothstep',
      }));
    }
    if (!edge.to) return [];
    return [{
      id: `${edgeIndex}:${edge.from}:${edge.to}`,
      source: edge.from,
      sourceHandle: 'default',
      target: edge.to,
      markerEnd: { type: MarkerType.ArrowClosed },
      data: { edgeIndex },
      type: 'smoothstep',
    }];
  }), [catalog.route_functions, state.edges]);

  useEffect(() => {
    setFlowNodes((current) => graphNodes.map((node) => {
      const existing = current.find((candidate) => candidate.id === node.id);
      if (!existing) return node;
      const savedNode = state.nodes.find((candidate) => candidate.id === node.id);
      const savedDecoration = [...(state.builder_ui?.notes || []), ...(state.builder_ui?.groups || [])]
        .find((candidate) => candidate.id === node.id);
      return {
        ...node,
        position: savedNode?.position || savedDecoration?.position || existing.position,
      };
    }));
  }, [graphNodes, state.builder_ui?.groups, state.builder_ui?.notes, state.nodes]);

  useEffect(() => {
    if (didInitialLayout.current || flowNodes.length === 0) return;
    didInitialLayout.current = true;
    if (state.nodes.length > 0 && state.nodes.every((node) => Boolean(node.position))) {
      const startTargetId = state.edges.find((edge) => edge.from === 'START')?.to;
      const endSourceId = state.edges.find((edge) => edge.to === 'END')?.from;
      const startTarget = state.nodes.find((node) => node.id === startTargetId);
      const endSource = state.nodes.find((node) => node.id === endSourceId);
      setFlowNodes((current) => current.map((node) => {
        if (node.id === 'START' && startTarget?.position) {
          return { ...node, position: { x: startTarget.position.x - WIDTH - 132, y: startTarget.position.y } };
        }
        if (node.id === 'END' && endSource?.position) {
          return { ...node, position: { x: endSource.position.x + WIDTH + 132, y: endSource.position.y } };
        }
        return node;
      }));
      window.requestAnimationFrame(() => fitView({ padding: 0.16 }));
      return;
    }
    void layout(flowNodes, graphEdges).then((next) => {
      setFlowNodes(next);
      window.requestAnimationFrame(() => fitView({ padding: 0.16 }));
    });
  }, [fitView, flowNodes.length, graphEdges, state.edges, state.nodes]);

  const isValidConnection = useCallback((connection: Connection | Edge) => (
    Boolean(connection.source && connection.target && canConnectNodes(catalog, state, connection.source, connection.target).ok)
  ), [catalog, state]);

  const beginConnection = useCallback((params: OnConnectStartParams) => {
    setConnectionMessage(null);
    setConnecting(params.nodeId ? {
      source: params.nodeId,
      route: params.handleId && params.handleId !== 'default' ? params.handleId : undefined,
    } : null);
  }, []);

  const finishConnection = useCallback((
    connectionState: FinalConnectionState,
    openPickerOnEmpty: boolean,
    droppedNodeId?: string,
  ) => {
    if (!connecting) return;
    if (connectionState.isValid) {
      setConnecting(null);
      return;
    }
    const targetId = connectionState.toNode?.id || droppedNodeId;
    if (targetId) {
      const result = canConnectNodes(catalog, state, connecting.source, targetId);
      setConnectionMessage(result.reason || 'These two steps cannot be connected.');
    } else if (openPickerOnEmpty) {
      onRequestAddNext(connecting.source, connecting.route);
    }
    setConnecting(null);
  }, [catalog, connecting, onRequestAddNext, state]);

  const tidy = async () => {
    const next = await layout(flowNodes, graphEdges);
    setFlowNodes(next);
    onPositionsChange(Object.fromEntries(next.map((node) => [node.id, node.position])));
    window.requestAnimationFrame(() => fitView({ padding: 0.16, duration: 300 }));
  };

  return (
    <ReactFlow
      nodes={flowNodes}
      edges={graphEdges}
      nodeTypes={nodeTypes}
      nodesConnectable={!disabled}
      nodesDraggable={!disabled}
      connectOnClick
      connectionRadius={36}
      autoPanOnConnect
      isValidConnection={isValidConnection}
      onNodesChange={(changes) => setFlowNodes((nodes) => applyNodeChanges(changes, nodes))}
      onNodeDragStop={(_, node) => {
        if ([...(state.builder_ui?.notes || []), ...(state.builder_ui?.groups || [])].some((item) => item.id === node.id)) onUpdateNote(node.id, node.position);
        else if (node.id !== 'START' && node.id !== 'END') onNodePositionChange(node.id, node.position);
      }}
      onConnect={(connection) => {
        if (!connection.source || !connection.target || !isValidConnection(connection)) {
          setConnectionMessage('These two steps cannot be connected.');
          return;
        }
        onConnectNodes(connection.source, connection.target, connection.sourceHandle && connection.sourceHandle !== 'default' ? connection.sourceHandle : undefined);
      }}
      onConnectStart={(_, params) => beginConnection(params)}
      onConnectEnd={(event, connectionState) => {
        const target = event.target as Element | null;
        const droppedNodeId = target?.closest?.('.react-flow__node')?.getAttribute('data-id') || undefined;
        finishConnection(connectionState, true, droppedNodeId);
      }}
      onClickConnectStart={(_, params) => beginConnection(params)}
      onClickConnectEnd={(event, connectionState) => {
        const target = event.target as Element | null;
        const droppedNodeId = target?.closest?.('.react-flow__node')?.getAttribute('data-id') || undefined;
        finishConnection(connectionState, false, droppedNodeId);
      }}
      onNodeClick={(_, node) => node.id !== 'START' && node.id !== 'END' && onSelectionChange({ kind: 'node', nodeId: node.id })}
      onEdgeClick={(_, edge) => onSelectionChange({ kind: 'edge', edgeIndex: Number(edge.data?.edgeIndex) })}
      onEdgesDelete={(edges) => edges
        .map((edge) => ({ index: Number(edge.data?.edgeIndex), route: edge.data?.route ? String(edge.data.route) : undefined }))
        .sort((a, b) => b.index - a.index)
        .forEach(({ index, route }) => onRemoveEdge(index, route))}
      onNodesDelete={(nodes) => nodes.filter((node) => node.id !== 'START' && node.id !== 'END').forEach((node) => {
        if ([...(state.builder_ui?.notes || []), ...(state.builder_ui?.groups || [])].some((item) => item.id === node.id)) onRemoveNote(node.id);
        else onRemoveNode(node.id);
      })}
      onPaneClick={() => {
        onSelectionChange(null);
        setConnectionMessage(null);
      }}
      onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; }}
      onDrop={(event) => {
        event.preventDefault();
        const nodeType = event.dataTransfer.getData('application/askpdf-node-type');
        if (!nodeType) return;
        onAddNodeAt(nodeType, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      }}
      fitView
      minZoom={0.35}
      maxZoom={1.8}
      deleteKeyCode={disabled ? null : ['Backspace', 'Delete']}
      onBeforeDelete={async ({ nodes, edges }) => {
        if (nodes.length === 0 && edges.length === 0) return true;
        return typeof window === 'undefined' || window.confirm(`Delete ${nodes.length + edges.length} selected graph element${nodes.length + edges.length === 1 ? '' : 's'}?`);
      }}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={24} size={1} />
      <Controls />
      <MiniMap pannable zoomable />
      {connecting ? (
        <Panel position="top-left">
          <Alert severity="info" variant="outlined" sx={{ py: 0 }}>
            Choose a compatible destination · Esc to cancel
          </Alert>
        </Panel>
      ) : connectionMessage ? (
        <Panel position="top-left">
          <Alert severity="warning" variant="outlined" onClose={() => setConnectionMessage(null)} sx={{ py: 0 }}>
            {connectionMessage}
          </Alert>
        </Panel>
      ) : null}
      <Panel position="top-right">
        <Button size="small" variant="outlined" startIcon={<AutoFixHighIcon />} onClick={tidy}>Tidy up</Button>
      </Panel>
    </ReactFlow>
  );
}

export default function WorkflowBuilderCanvas(props: React.ComponentProps<typeof Canvas>) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}
