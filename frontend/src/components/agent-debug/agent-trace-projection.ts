import type { AgentDebugTrace, AgentRunDebug, AgentRunDetails } from '../../lib/api';
import type { AgentGraphEdge, AgentGraphNode } from '../agent-graph/agent-graph-types';

export interface TraceNodeView {
  id: string;
  status?: string;
  skipped: boolean;
  durationMs?: number;
  route?: string;
  routeReason?: string;
  executionPlan?: string[];
  warningCodes: string[];
  error?: Record<string, any>;
  span?: Record<string, any>;
  raw: Record<string, any>;
}

export interface TraceToolView {
  name: string;
  id?: string;
  category?: string;
  displayName?: string;
  callerNode?: string;
  ok: boolean;
  durationMs?: number;
  sourceCount?: number;
  warningCodes: string[];
  span?: Record<string, any>;
  raw: Record<string, any>;
}

export interface TraceGraphView {
  nodes: AgentGraphNode[];
  edges: AgentGraphEdge[];
  executionPlan: string[];
  selectedRoute?: string;
}

export interface TraceRunView {
  debug?: AgentRunDebug;
  trace?: AgentDebugTrace;
  graph?: TraceGraphView;
  route?: string;
  routeReason?: string;
  metrics: Record<string, any>;
  nodes: TraceNodeView[];
  tools: TraceToolView[];
  usedNodeCount: number;
  availableNodeCount?: number;
  usedToolCount: number;
  availableToolCount?: number;
  warningCount: number;
  errorCount: number;
  errors: Record<string, any>[];
}

const asObject = (value: any): Record<string, any> => (
  value && typeof value === 'object' && !Array.isArray(value) ? value : {}
);

const asArray = (value: any): Record<string, any>[] => (
  Array.isArray(value) ? value.filter((item): item is Record<string, any> => item && typeof item === 'object') : []
);

const asStringArray = (value: any): string[] => (
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
);

const asNumber = (value: any): number | undefined => {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
};

export const getRunDebug = (runDetails: AgentRunDetails): AgentRunDebug | undefined => {
  const debug = runDetails.debug;
  if (!debug || typeof debug !== 'object' || Array.isArray(debug)) return undefined;
  if (debug.version !== 1) return undefined;
  if (Object.keys(asObject(debug.trace)).length === 0) return undefined;
  if (Object.keys(asObject(debug.summary)).length === 0) return undefined;
  return debug;
};

export const getRunTrace = (runDetails: AgentRunDetails): AgentDebugTrace | undefined => {
  const trace = getRunDebug(runDetails)?.trace;
  return trace && typeof trace === 'object' && !Array.isArray(trace) ? trace : undefined;
};

export const getRunDebugMetrics = (runDetails: AgentRunDetails) => {
  const debug = getRunDebug(runDetails);
  const summaryMetrics = asObject(debug?.summary?.metrics);
  if (Object.keys(summaryMetrics).length > 0) return summaryMetrics;
  const traceMetrics = asObject(debug?.trace?.metrics);
  if (Object.keys(traceMetrics).length > 0) return traceMetrics;
  return runDetails.metrics_json || {};
};

const nodeViewFromSummary = (row: Record<string, any>): TraceNodeView => ({
  id: String(row.id || row.node || row.name || 'unknown_node'),
  status: typeof row.status === 'string' ? row.status : undefined,
  skipped: row.skipped === true || row.status === 'skipped',
  durationMs: asNumber(row.durationMs ?? row.duration_ms),
  route: typeof row.route === 'string' ? row.route : undefined,
  routeReason: typeof row.routeReason === 'string' ? row.routeReason : typeof row.route_reason === 'string' ? row.route_reason : undefined,
  executionPlan: asStringArray(row.executionPlan ?? row.execution_plan),
  warningCodes: asStringArray(row.warningCodes ?? row.warnings),
  error: row.error && typeof row.error === 'object' ? row.error : undefined,
  span: row.span && typeof row.span === 'object' ? row.span : undefined,
  raw: asObject(row.raw),
});

const toolViewFromSummary = (row: Record<string, any>): TraceToolView => ({
  name: String(row.name || row.tool_name || 'tool'),
  id: typeof row.id === 'string' ? row.id : typeof row.tool_id === 'string' ? row.tool_id : undefined,
  category: typeof row.category === 'string' ? row.category : typeof row.tool_category === 'string' ? row.tool_category : undefined,
  displayName: typeof row.displayName === 'string' ? row.displayName : typeof row.tool_display_name === 'string' ? row.tool_display_name : undefined,
  callerNode: typeof row.callerNode === 'string' ? row.callerNode : typeof row.caller_node === 'string' ? row.caller_node : undefined,
  ok: row.ok !== false,
  durationMs: asNumber(row.durationMs ?? row.elapsed_ms),
  sourceCount: asNumber(row.sourceCount ?? row.source_count),
  warningCodes: asStringArray(row.warningCodes ?? row.warnings),
  span: row.span && typeof row.span === 'object' ? row.span : undefined,
  raw: asObject(row.raw),
});

const getRunGraph = (debug?: AgentRunDebug): TraceGraphView | undefined => {
  const graph = asObject(debug?.graph);
  const nodes = asArray(graph.nodes) as AgentGraphNode[];
  const edges = asArray(graph.edges) as AgentGraphEdge[];
  if (nodes.length === 0 && edges.length === 0) return undefined;
  return {
    nodes,
    edges,
    executionPlan: asStringArray(graph.executionPlan),
    selectedRoute: typeof graph.selectedRoute === 'string' ? graph.selectedRoute : undefined,
  };
};

export const buildRunTraceView = (runDetails: AgentRunDetails): TraceRunView | undefined => {
  const debug = getRunDebug(runDetails);
  if (!debug) return undefined;
  const summary = asObject(debug.summary);
  const metrics = getRunDebugMetrics(runDetails);
  const nodes = asArray(summary.nodes).map(nodeViewFromSummary);
  const tools = asArray(summary.tools).map(toolViewFromSummary);
  const usedNodeCount = asNumber(summary.usedNodeCount) ?? nodes.filter((node) => !node.skipped).length;
  const usedToolCount = asNumber(summary.usedToolCount) ?? tools.length;
  return {
    debug,
    trace: getRunTrace(runDetails),
    graph: getRunGraph(debug),
    route: typeof summary.route === 'string' ? summary.route : typeof metrics.route === 'string' ? metrics.route : undefined,
    routeReason: typeof summary.routeReason === 'string' ? summary.routeReason : undefined,
    metrics,
    nodes,
    tools,
    usedNodeCount,
    availableNodeCount: asNumber(summary.availableNodeCount),
    usedToolCount,
    availableToolCount: asNumber(summary.availableToolCount),
    warningCount: asNumber(summary.warningCount) ?? Number(metrics.tool_warning_count ?? 0),
    errorCount: asNumber(summary.errorCount) ?? Number(metrics.error_count ?? metrics.tool_error_count ?? 0),
    errors: asArray(summary.errors),
  };
};

export const buildTraceExportJson = (view?: TraceRunView): string => (
  view?.debug ? JSON.stringify(view.debug, null, 2) : ''
);
