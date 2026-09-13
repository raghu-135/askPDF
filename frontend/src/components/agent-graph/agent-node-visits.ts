import type { TraceOperationView } from '../agent-debug/agent-trace-projection';
import type { AgentGraphEdge, AgentGraphNode, AgentNodeVisitRef } from './agent-graph-types';

export const normalizeVisitIndex = (value: unknown): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 1 ? Math.floor(numeric) : 1;
};

export const toAgentNodeVisitRef = (
  node: Pick<TraceOperationView, 'id' | 'visitIndex'>,
): AgentNodeVisitRef => ({
  nodeId: node.id,
  visitIndex: normalizeVisitIndex(node.visitIndex),
});

export const agentNodeVisitKey = (
  visit: AgentNodeVisitRef | Pick<TraceOperationView, 'id' | 'visitIndex'>,
): string => {
  const nodeId = 'nodeId' in visit ? visit.nodeId : visit.id;
  return `${nodeId}:${normalizeVisitIndex(visit.visitIndex)}`;
};

export const getChronologicalNodeVisits = (
  nodes: readonly TraceOperationView[],
  nodeId: string,
): TraceOperationView[] => nodes.filter((node) => node.id === nodeId);

export const getLatestNodeVisit = (
  nodes: readonly TraceOperationView[],
  nodeId: string,
): TraceOperationView | undefined => {
  const visits = getChronologicalNodeVisits(nodes, nodeId);
  return visits[visits.length - 1];
};

const getAdjacentNodeVisit = (
  nodes: readonly TraceOperationView[],
  selected: AgentNodeVisitRef,
  offset: -1 | 1,
): TraceOperationView | undefined => {
  const visits = getChronologicalNodeVisits(nodes, selected.nodeId);
  const selectedKey = agentNodeVisitKey(selected);
  const selectedPosition = visits.findIndex((visit) => agentNodeVisitKey(visit) === selectedKey);
  if (selectedPosition < 0) return undefined;
  return visits[selectedPosition + offset];
};

export const getPreviousNodeVisit = (
  nodes: readonly TraceOperationView[],
  selected: AgentNodeVisitRef,
): TraceOperationView | undefined => getAdjacentNodeVisit(nodes, selected, -1);

export const getNextNodeVisit = (
  nodes: readonly TraceOperationView[],
  selected: AgentNodeVisitRef,
): TraceOperationView | undefined => getAdjacentNodeVisit(nodes, selected, 1);

const nonEmptyString = (value: unknown): string | undefined => (
  typeof value === 'string' && value.length > 0 ? value : undefined
);

/** Returns the route chosen by this invocation, including evaluator-specific routes. */
export const getNodeVisitRoute = (node: TraceOperationView | undefined): string | undefined => {
  if (!node) return undefined;
  const raw = node.raw || {};
  const detail = raw.detail && typeof raw.detail === 'object' ? raw.detail : {};
  const event = detail.event && typeof detail.event === 'object' ? detail.event : {};
  return nonEmptyString(event.evaluator_route ?? event.evaluatorRoute)
    ?? nonEmptyString(raw.evaluator_route ?? raw.evaluatorRoute)
    ?? nonEmptyString(node.route)
    ?? nonEmptyString(event.route)
    ?? nonEmptyString(raw.route);
};

/** Applies only the selected invocation to its logical node and outgoing route. */
export const applySelectedVisitOverlay = <T extends { nodes: AgentGraphNode[]; edges: AgentGraphEdge[] }>(
  graph: T,
  traceNodes: readonly TraceOperationView[],
  selected: AgentNodeVisitRef | null | undefined,
): T => {
  if (!selected) return graph;
  const topologyVisits = traceNodes.filter((node) => String(node.topologyRef?.id || node.id) === selected.nodeId);
  const selectedRow = topologyVisits.find((node) => normalizeVisitIndex(node.visitIndex) === selected.visitIndex);
  const selectedPosition = topologyVisits.findIndex((visit) => visit === selectedRow);
  const route = getNodeVisitRoute(selectedRow);
  return {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === selected.nodeId ? {
      ...node,
      selectedVisitIndex: selected.visitIndex,
      selectedVisitPosition: selectedPosition >= 0 ? selectedPosition + 1 : undefined,
    } : node),
    edges: graph.edges.map((edge) => edge.source === selected.nodeId && edge.conditional && route ? {
      ...edge,
      selected: edge.route === route,
    } : edge),
  };
};
