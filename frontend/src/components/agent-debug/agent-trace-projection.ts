import type { AgentDebugTrace, AgentRunDebug, AgentRunDetails, AgentRunFinalOutput, AgentRunNodeDetailManifest, BuilderTestStreamEnvelope } from '../../lib/api';
import {
  formatNodeInstanceLabel,
  formatNodeLabel,
} from '../agent-graph/agent-node-labels.js';
import type { AgentGraphEdge, AgentGraphNode, AgentNodeCatalog } from '../agent-graph/agent-graph-types';

export interface TraceNodeView {
  id: string;
  type?: string;
  label: string;
  instanceLabel: string;
  visitIndex?: number;
  status?: string;
  skipped: boolean;
  durationMs?: number;
  route?: string;
  routeReason?: string;
  executionPlan?: string[];
  usedMemoryIdCount?: number;
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
  callerNodeType?: string;
  callerVisitIndex?: number;
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
  memory?: {
    recalledMemoryIds: string[];
    searchedScopes: Record<string, any>[];
    recalledCount: number;
  };
  finalOutput?: AgentRunFinalOutput;
  detailManifest: AgentRunNodeDetailManifest[];
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

const asOptionalStringArray = (value: any): string[] | undefined => {
  const items = asStringArray(value).filter((item) => item.length > 0);
  return items.length > 0 ? items : undefined;
};

const asNumber = (value: any): number | undefined => {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
};

const asNonEmptyString = (value: any): string | undefined => (
  typeof value === 'string' && value.length > 0 ? value : undefined
);

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

const nodeViewFromSummary = (row: Record<string, any>, nodeCatalog?: AgentNodeCatalog): TraceNodeView => {
  const raw = asObject(row.raw);
  const id = String(row.id || row.node || row.name || raw.node || 'unknown_node');
  const type = typeof row.type === 'string'
    ? row.type
    : typeof row.node_type === 'string'
      ? row.node_type
      : typeof raw.node_type === 'string'
        ? raw.node_type
        : undefined;
  return {
    id,
    type,
    label: formatNodeLabel(id, type, nodeCatalog)
      || asNonEmptyString(row.label)
      || asNonEmptyString(row.node_name)
      || asNonEmptyString(raw.label)
      || asNonEmptyString(raw.node_name),
    instanceLabel: formatNodeInstanceLabel(id, type),
    visitIndex: asNumber(row.visitIndex ?? row.visit_index ?? raw.visit_index ?? raw.visitIndex),
    status: typeof row.status === 'string' ? row.status : undefined,
    skipped: row.skipped === true || row.status === 'skipped',
    durationMs: asNumber(row.durationMs ?? row.duration_ms),
    route: typeof row.route === 'string' ? row.route : undefined,
    routeReason: typeof row.routeReason === 'string' ? row.routeReason : typeof row.route_reason === 'string' ? row.route_reason : undefined,
    executionPlan: asStringArray(row.executionPlan ?? row.execution_plan),
    usedMemoryIdCount: asNumber(row.usedMemoryIdCount ?? row.used_memory_id_count ?? raw.used_memory_id_count),
    warningCodes: asStringArray(row.warningCodes ?? row.warnings),
    error: row.error && typeof row.error === 'object' ? row.error : undefined,
    span: row.span && typeof row.span === 'object' ? row.span : undefined,
    raw,
  };
};

const toolViewFromSummary = (row: Record<string, any>): TraceToolView => {
  const raw = asObject(row.raw);
  return {
    name: String(row.name || row.tool_name || raw.tool_name || 'tool'),
    id: typeof row.id === 'string' ? row.id : typeof row.tool_id === 'string' ? row.tool_id : undefined,
    category: typeof row.category === 'string' ? row.category : typeof row.tool_category === 'string' ? row.tool_category : undefined,
    displayName: typeof row.displayName === 'string' ? row.displayName : typeof row.tool_display_name === 'string' ? row.tool_display_name : undefined,
    callerNode: typeof row.callerNode === 'string' ? row.callerNode : typeof row.caller_node === 'string' ? row.caller_node : typeof raw.caller_node === 'string' ? raw.caller_node : undefined,
    callerNodeType: typeof row.callerNodeType === 'string'
      ? row.callerNodeType
      : typeof row.caller_node_type === 'string'
        ? row.caller_node_type
        : typeof raw.caller_node_type === 'string'
          ? raw.caller_node_type
          : undefined,
    callerVisitIndex: asNumber(row.callerVisitIndex ?? row.caller_visit_index ?? raw.caller_visit_index ?? raw.callerVisitIndex),
    ok: row.ok !== false,
    durationMs: asNumber(row.durationMs ?? row.elapsed_ms),
    sourceCount: asNumber(row.sourceCount ?? row.source_count),
    warningCodes: asStringArray(row.warningCodes ?? row.warnings),
    span: row.span && typeof row.span === 'object' ? row.span : undefined,
    raw,
  };
};

const getRunGraph = (debug?: AgentRunDebug, nodeCatalog?: AgentNodeCatalog): TraceGraphView | undefined => {
  const graph = asObject(debug?.graph);
  const nodes = (asArray(graph.nodes) as AgentGraphNode[]).map((node) => {
    const id = String(node.id || 'unknown_node');
    const type = typeof node.type === 'string' ? node.type : id;
    const catalogEntry = asObject(nodeCatalog?.[type]);
    const category = typeof node.category === 'string'
      ? node.category
      : typeof catalogEntry.category === 'string'
        ? catalogEntry.category
        : undefined;
    return {
      ...node,
      id,
      type,
      label: formatNodeLabel(id, type, nodeCatalog) || asNonEmptyString(node.label) || id,
      category,
      capabilities: asOptionalStringArray(node.capabilities) || asOptionalStringArray(catalogEntry.capabilities),
      observability: asObject(node.observability) || asObject(catalogEntry.observability),
      instanceId: id,
      instanceLabel: formatNodeInstanceLabel(id, type),
    };
  });
  const edges = asArray(graph.edges) as AgentGraphEdge[];
  if (nodes.length === 0 && edges.length === 0) return undefined;
  return {
    nodes,
    edges,
    executionPlan: asStringArray(graph.executionPlan),
    selectedRoute: typeof graph.selectedRoute === 'string' ? graph.selectedRoute : undefined,
  };
};

export const buildRunTraceView = (
  runDetails: AgentRunDetails,
  options: { nodeCatalog?: AgentNodeCatalog } = {},
): TraceRunView | undefined => {
  try {
    const debug = getRunDebug(runDetails);
    if (!debug) return undefined;
    const summary = asObject(debug.summary);
    const metrics = getRunDebugMetrics(runDetails);
    const nodes = asArray(summary.nodes).map((node) => nodeViewFromSummary(node, options.nodeCatalog));
    const tools = asArray(summary.tools).map(toolViewFromSummary);
    const usedNodeCount = asNumber(summary.usedNodeCount) ?? nodes.filter((node) => !node.skipped).length;
    const usedToolCount = asNumber(summary.usedToolCount) ?? tools.length;
    const memory = asObject(summary.memory);
    return {
      debug,
      trace: getRunTrace(runDetails),
      graph: getRunGraph(debug, options.nodeCatalog),
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
      memory: Object.keys(memory).length > 0 ? {
        recalledMemoryIds: asStringArray(memory.recalledMemoryIds),
        searchedScopes: asArray(memory.searchedScopes),
        recalledCount: asNumber(memory.recalledCount) ?? 0,
      } : undefined,
      finalOutput: runDetails.final_output || debug.final_output,
      detailManifest: Array.isArray(debug.detail_manifest) ? debug.detail_manifest : [],
    };
  } catch (err) {
    if (typeof console !== 'undefined') {
      console.error('Unable to project agent trace payload', err);
    }
    return undefined;
  }
};

export const buildLiveTraceView = (
  events: BuilderTestStreamEnvelope[],
): TraceRunView => {
  const nodes: TraceNodeView[] = [];
  const nodeIndex = new Map<string, number>();
  const tools: TraceToolView[] = [];
  let finalOutput: AgentRunFinalOutput | undefined;
  let route: string | undefined;
  let routeReason: string | undefined;
  const runErrors: Record<string, any>[] = [];

  events.forEach((envelope) => {
    const data = asObject(envelope.data);
    if (envelope.event.startsWith('node.') && typeof data.node_id === 'string') {
      const visitIndex = asNumber(data.visit_index) || 1;
      const key = `${data.node_id}:${visitIndex}`;
      const status = envelope.event === 'node.started' ? 'active'
        : envelope.event === 'node.failed' ? 'error'
          : envelope.event === 'node.skipped' ? 'skipped'
            : 'completed';
      const rawError = data.detail?.error ?? data.error;
      const row: TraceNodeView = {
        id: data.node_id,
        type: asNonEmptyString(data.node_type),
        label: formatNodeLabel(data.node_id, asNonEmptyString(data.node_type)),
        instanceLabel: formatNodeInstanceLabel(data.node_id, asNonEmptyString(data.node_type)),
        visitIndex,
        status,
        skipped: status === 'skipped',
        route: asNonEmptyString(data.route),
        routeReason: asNonEmptyString(data.route_reason),
        warningCodes: asStringArray(data.warnings),
        error: typeof rawError === 'object' && rawError !== null
          ? asObject(rawError)
          : rawError ? { raw_message: String(rawError) } : {},
        raw: data,
      };
      const existing = nodeIndex.get(key);
      if (existing === undefined) {
        nodeIndex.set(key, nodes.length);
        nodes.push(row);
      } else {
        nodes[existing] = { ...nodes[existing], ...row, raw: { ...nodes[existing].raw, ...data } };
      }
      route = row.route || route;
      routeReason = row.routeReason || routeReason;
    }
    if (envelope.event === 'tool.completed') tools.push(toolViewFromSummary(data));
    if (envelope.event === 'run.completed') {
      finalOutput = asObject(data.final_output) as AgentRunFinalOutput;
      if (!finalOutput.answer && typeof data.answer === 'string') finalOutput.answer = data.answer;
      route = asNonEmptyString(data.route) || route;
      routeReason = asNonEmptyString(data.route_reason) || routeReason;
    }
    if (envelope.event === 'run.failed') {
      const rawError = data.error;
      runErrors.push(
        rawError && typeof rawError === 'object'
          ? asObject(rawError)
          : { raw_message: String(rawError || data.message || 'Workflow test failed.') },
      );
    }
  });

  return {
    route,
    routeReason,
    metrics: {},
    nodes,
    tools,
    usedNodeCount: new Set(nodes.filter((node) => !node.skipped).map((node) => node.id)).size,
    usedToolCount: tools.length,
    warningCount: nodes.reduce((count, node) => count + node.warningCodes.length, 0) + tools.reduce((count, tool) => count + tool.warningCodes.length, 0),
    errorCount: nodes.filter((node) => node.status === 'error').length + tools.filter((tool) => !tool.ok).length + runErrors.length,
    errors: [
      ...nodes.map((node) => node.error).filter((error): error is Record<string, any> => Boolean(error && Object.keys(error).length)),
      ...runErrors,
    ],
    finalOutput,
    detailManifest: nodes.filter((node) => node.raw.detail).map((node) => ({
      node_id: node.id,
      node_type: node.type,
      visit_index: node.visitIndex || 1,
      status: node.status,
      available: true,
      truncated: Boolean(node.raw.detail?.safety?.truncated),
    })),
  };
};

export const mergeLiveAndRetainedTraceViews = (
  live: TraceRunView,
  retained?: TraceRunView,
): TraceRunView => {
  if (!retained) return live;
  const liveVisits = new Set(live.nodes.map((node) => `${node.id}:${node.visitIndex || 1}`));
  const nodes = [
    ...retained.nodes.filter((node) => !liveVisits.has(`${node.id}:${node.visitIndex || 1}`)),
    ...live.nodes,
  ];
  const toolKeys = new Set(live.tools.map((tool, index) => `${tool.id || tool.name}:${tool.callerNode || ''}:${tool.callerVisitIndex || 1}:${index}`));
  const retainedTools = retained.tools.filter((tool, index) => !toolKeys.has(`${tool.id || tool.name}:${tool.callerNode || ''}:${tool.callerVisitIndex || 1}:${index}`));
  const tools = [...retainedTools, ...live.tools];
  const detailManifest = new Map(
    [...retained.detailManifest, ...live.detailManifest]
      .map((row) => [`${row.node_id}:${row.visit_index}`, row] as const),
  );
  return {
    ...retained,
    ...live,
    graph: retained.graph || live.graph,
    route: live.route || retained.route,
    routeReason: live.routeReason || retained.routeReason,
    nodes,
    tools,
    usedNodeCount: new Set(nodes.filter((node) => !node.skipped).map((node) => node.id)).size,
    usedToolCount: tools.length,
    warningCount: nodes.reduce((count, node) => count + node.warningCodes.length, 0)
      + tools.reduce((count, tool) => count + tool.warningCodes.length, 0),
    errorCount: nodes.filter((node) => node.status === 'error').length
      + tools.filter((tool) => !tool.ok).length,
    errors: nodes.map((node) => node.error).filter((nodeError): nodeError is Record<string, any> => Boolean(nodeError && Object.keys(nodeError).length)),
    finalOutput: live.finalOutput || retained.finalOutput,
    detailManifest: [...detailManifest.values()],
  };
};

export const buildTraceExportJson = (view?: TraceRunView): string => (
  view?.debug ? JSON.stringify(view.debug, null, 2) : ''
);
