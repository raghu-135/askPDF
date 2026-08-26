import type { AgentDebugTrace, AgentRunDebug, AgentRunDetails, AgentRunFinalOutput, AgentRunOperationDetailManifest, AgentTraceDiagnostics, AgentTraceFailure, AgentTraceLocation, AgentTraceModelInvocation, AgentTraceParallelGroup, AgentTraceTimelineEvent, AgentTraceVisualization, BuilderTestStreamEnvelope } from '../../lib/api';
import {
  formatNodeInstanceLabel,
  formatNodeLabel,
} from '../agent-graph/agent-node-labels.js';
import type { AgentGraphEdge, AgentGraphNode, AgentNodeCatalog } from '../agent-graph/agent-graph-types';
import { normalizeAgentExecutionStatus } from '../agent-graph/agent-execution-status.ts';

export function getRetainedRunErrorMessage(runDetails: { error_json?: Record<string, any> | null }): string | null {
  const error = runDetails.error_json;
  if (typeof error?.safe_message === 'string' && error.safe_message.trim()) return error.safe_message;
  if (typeof error?.message === 'string' && error.message.trim()) return error.message;
  return null;
}

export function shouldRefreshRetainedTrace(runDetails: AgentRunDetails): boolean {
  const terminal = ['completed', 'failed', 'cancelled'].includes(String(runDetails.status));
  return terminal && !runDetails.debug;
}

export interface TraceOperationView {
  id: string;
  type?: string;
  label: string;
  instanceLabel: string;
  parentOperationId?: string;
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
  topologyRef?: { kind?: string; id?: string; [key: string]: any };
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
  status?: string;
  durationMs?: number;
  sourceCount?: number;
  warningCodes: string[];
  span?: Record<string, any>;
  raw: Record<string, any>;
}

export interface TraceModelView extends AgentTraceModelInvocation {
  raw: Record<string, any>;
}

export interface TraceGraphView {
  nodes: AgentGraphNode[];
  edges: AgentGraphEdge[];
  executionPlan: string[];
  selectedRoute?: string;
}

export interface TraceRunView {
  parseError?: string;
  parseCorrelationId?: string;
  debug?: AgentRunDebug;
  trace?: AgentDebugTrace;
  graph?: TraceGraphView;
  route?: string;
  routeReason?: string;
  metrics: Record<string, any>;
  events: AgentTraceTimelineEvent[];
  visualizations: Record<string, AgentTraceVisualization>;
  operations: TraceOperationView[];
  tools: TraceToolView[];
  models: TraceModelView[];
  usedOperationCount: number;
  availableOperationCount?: number;
  usedToolCount: number;
  availableToolCount?: number;
  warningCount: number;
  errorCount: number;
  diagnostics: AgentTraceDiagnostics;
  parallelGroups: AgentTraceParallelGroup[];
  memory?: {
    recalledMemoryIds: string[];
    searchedScopes: Record<string, any>[];
    recalledCount: number;
  };
  finalOutput?: AgentRunFinalOutput;
  detailManifest: AgentRunOperationDetailManifest[];
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

export type AgentRunDebugParseResult =
  | { ok: true; debug: AgentRunDebug }
  | { ok: false; reason: string; correlationId: string };

const parseFailure = (runId: string, reason: string): AgentRunDebugParseResult => ({
  ok: false,
  reason: reason.slice(0, 240),
  correlationId: `trace:${runId}`,
});

const validStringArray = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === 'string');

const validateParallelGroups = (
  value: unknown,
  events: unknown,
  { requireEventReferences = true }: { requireEventReferences?: boolean } = {},
): string | null => {
  if (!Array.isArray(value)) return 'parallel_groups must be an array.';
  if (!Array.isArray(events)) return 'Canonical events must be an array.';
  const eventIds = new Set(events.map((event) => asNonEmptyString(asObject(event).event_id)).filter(Boolean));
  const validReferences = (references: unknown) => (
    validStringArray(references)
    && (!requireEventReferences || references.every((eventId) => eventIds.has(eventId)))
  );
  const groupIds = new Set<string>();
  const memberOwners = new Map<string, string>();
  for (const group of value) {
    if (!group || typeof group !== 'object' || Array.isArray(group)) return 'A parallel group is not an object.';
    const row = group as Record<string, any>;
    if (typeof row.group_id !== 'string' || !row.group_id) return 'A parallel group is missing group_id.';
    if (groupIds.has(row.group_id)) return `Parallel group ${row.group_id} is duplicated.`;
    groupIds.add(row.group_id);
    if (typeof row.status !== 'string' || !Number.isInteger(row.planned) || row.planned < 0) return `Parallel group ${row.group_id} has invalid status or planned count.`;
    if (!Number.isInteger(row.first_sequence) || !Number.isInteger(row.last_sequence) || row.last_sequence < row.first_sequence) return `Parallel group ${row.group_id} has invalid sequence data.`;
    if (!validReferences(row.event_ids) || !Array.isArray(row.members)) return `Parallel group ${row.group_id} has invalid event or member data.`;
    if (!row.barrier || typeof row.barrier !== 'object' || typeof row.barrier.status !== 'string') return `Parallel group ${row.group_id} has invalid barrier data.`;
    if (!row.aggregation || typeof row.aggregation !== 'object' || typeof row.aggregation.status !== 'string' || !row.aggregation.counts || typeof row.aggregation.counts !== 'object') return `Parallel group ${row.group_id} has invalid aggregation data.`;
    if (Object.values(row.aggregation.counts).some((count) => !Number.isInteger(count) || Number(count) < 0)) return `Parallel group ${row.group_id} has invalid aggregation counts.`;
    for (const member of row.members) {
      if (!member || typeof member !== 'object' || Array.isArray(member)) return `Parallel group ${row.group_id} contains an invalid member.`;
      const memberRow = member as Record<string, any>;
      if (typeof memberRow.member_id !== 'string' || !memberRow.member_id) return `Parallel group ${row.group_id} contains a member without member_id.`;
      const owner = memberOwners.get(memberRow.member_id);
      if (owner && owner !== row.group_id) return `Parallel member ${memberRow.member_id} belongs to conflicting groups.`;
      memberOwners.set(memberRow.member_id, row.group_id);
      if (typeof memberRow.status !== 'string' || !Number.isInteger(memberRow.first_sequence) || !Number.isInteger(memberRow.last_sequence)) return `Parallel member ${memberRow.member_id} has invalid status or sequence data.`;
      if (!validReferences(memberRow.event_ids) || !Array.isArray(memberRow.attempts)) return `Parallel member ${memberRow.member_id} has invalid event or attempt data.`;
      for (const attempt of memberRow.attempts) {
        if (!attempt || typeof attempt !== 'object' || !Number.isInteger(attempt.attempt) || attempt.attempt < 1) return `Parallel member ${memberRow.member_id} has an invalid attempt.`;
        if (typeof attempt.status !== 'string' || !Number.isInteger(attempt.first_sequence) || !Number.isInteger(attempt.last_sequence)) return `Parallel member ${memberRow.member_id} has invalid attempt state.`;
        if (!validReferences(attempt.event_ids) || !validReferences(attempt.failure_event_ids) || !validStringArray(attempt.caused_by_event_ids) || !validStringArray(attempt.related_event_ids)) return `Parallel member ${memberRow.member_id} has invalid attempt references.`;
      }
    }
  }
  return null;
};

export const parseRunDebug = (runDetails: AgentRunDetails): AgentRunDebugParseResult => {
  const debug = runDetails.debug;
  if (!debug || typeof debug !== 'object' || Array.isArray(debug)) return parseFailure(runDetails.id, 'The debug payload is missing.');
  if (debug.version !== 1) return parseFailure(runDetails.id, 'The trace marker is not supported.');
  if (Object.keys(asObject(debug.diagnostics)).length === 0) return parseFailure(runDetails.id, 'The diagnostics contract is missing.');
  if (!Array.isArray(debug.events) || !Array.isArray(debug.operations)) return parseFailure(runDetails.id, 'Canonical events or operations are missing.');
  const parallelError = validateParallelGroups(debug.parallel_groups, debug.events);
  if (parallelError) return parseFailure(runDetails.id, parallelError);
  if (
    !Array.isArray(debug.tools)
    || !Array.isArray(debug.approvals)
    || !Array.isArray(debug.subagents)
    || !Array.isArray(debug.artifacts)
    || !Array.isArray(debug.details)
  ) return parseFailure(runDetails.id, 'Trace detail or resource collections are missing.');
  if (!debug.visualizations || typeof debug.visualizations !== 'object' || Array.isArray(debug.visualizations)) {
    return parseFailure(runDetails.id, 'The visualization contract is missing.');
  }
  if (Object.keys(asObject(debug.trace)).length === 0) return parseFailure(runDetails.id, 'The canonical trace is missing.');
  if (Object.keys(asObject(debug.summary)).length === 0) return parseFailure(runDetails.id, 'The trace summary is missing.');
  return { ok: true, debug: debug as AgentRunDebug };
};

export const getRunDebug = (runDetails: AgentRunDetails): AgentRunDebug | undefined => {
  const result = parseRunDebug(runDetails);
  return result.ok ? result.debug : undefined;
};

const diagnosticLocation = (event: AgentTraceTimelineEvent): AgentTraceLocation => {
  const payload = event.payload || {};
  return Object.fromEntries(Object.entries({
    operation_id: event.operation_id || payload.operation_id,
    operation_label: payload.operation_label || payload.label,
    parent_operation_id: event.parent_operation_id || payload.parent_operation_id || payload.parent_id,
    tool_call_id: payload.tool_call_id,
    tool_name: payload.tool_name,
    subagent_id: payload.subagent_id,
    approval_id: payload.approval_id,
    parallel_group_id: payload.parallel_group_id ?? payload.dispatch_id ?? (payload.wave_id !== undefined ? String(payload.wave_id) : undefined),
    attempt: event.attempt,
    sequence: event.sequence,
    topology_ref: payload.topology_ref,
  }).filter(([, value]) => value !== undefined && value !== null && value !== '')) as AgentTraceLocation;
};

export const buildDiagnosticsFromTimeline = (events: AgentTraceTimelineEvent[]): AgentTraceDiagnostics => {
  const failures: AgentTraceFailure[] = [];
  let terminal: AgentTraceFailure | undefined;
  events.forEach((event) => {
    const payload = event.payload || {};
    const status = String(event.status || payload.status || '').toLowerCase();
    const rawErrorValue = payload.error;
    const rawError = asObject(rawErrorValue);
    const failed = event.kind === 'run.failed' || event.kind.endsWith('.failed') || ['failed', 'failure', 'error', 'rejected'].includes(status) || Boolean(rawErrorValue) || (event.kind === 'tool.completed' && payload.ok === false);
    const cancelled = event.kind.endsWith('.cancelled') || ['cancelled', 'canceled'].includes(status);
    if (!failed && !cancelled) return;
    const causedBy = asNonEmptyString(payload.caused_by_event_id) || asNonEmptyString(rawError.caused_by_event_id);
    const row: AgentTraceFailure = {
      event_id: event.event_id,
      kind: event.kind,
      classification: ['run.failed', 'run.cancelled'].includes(event.kind) ? 'terminal_summary' : cancelled ? 'cancellation' : 'contributing',
      code: String(rawError.code || payload.code || event.kind.replaceAll('.', '_')),
      message: String(rawError.safe_message || rawError.message || rawError.raw_message || (typeof rawErrorValue === 'string' ? rawErrorValue : '') || payload.message || payload.reason || event.kind),
      retryable: Boolean(rawError.retryable || payload.retryable),
      occurred_at: event.occurred_at,
      location: diagnosticLocation(event),
      ...(causedBy ? { caused_by_event_id: causedBy } : {}),
      ...(Array.isArray(payload.related_event_ids) ? { related_event_ids: payload.related_event_ids.map(String) } : {}),
      ...(Object.keys(asObject(rawError.details)).length > 0 ? { details: asObject(rawError.details) } : {}),
    };
    failures.push(row);
    if (row.classification === 'terminal_summary') terminal = row;
  });
  const byId = new Map(failures.map((failure) => [failure.event_id, failure]));
  const nonTerminal = failures.filter((failure) => !['terminal_summary', 'cancellation'].includes(failure.classification));
  const explicitId = terminal?.caused_by_event_id;
  let primary = explicitId ? byId.get(explicitId) : undefined;
  const visited = new Set<string>();
  while (primary?.caused_by_event_id && !visited.has(primary.event_id)) {
    visited.add(primary.event_id);
    const next = byId.get(primary.caused_by_event_id);
    if (!next) break;
    primary = next;
  }
  primary ||= nonTerminal[0] || terminal;
  const primaryBasis = explicitId && primary ? 'explicit_cause' : primary ? 'earliest_observed' : null;
  const parallelCounts = new Map<string, number>();
  nonTerminal.forEach((failure) => {
    const groupId = failure.location.parallel_group_id;
    if (groupId) parallelCounts.set(groupId, (parallelCounts.get(groupId) || 0) + 1);
  });
  failures.forEach((failure) => {
    if (failure === primary && failure.classification !== 'terminal_summary') failure.classification = 'primary';
    else if (!['terminal_summary', 'cancellation'].includes(failure.classification)) {
      const groupId = failure.location.parallel_group_id;
      failure.classification = failure.caused_by_event_id ? 'downstream' : groupId && (parallelCounts.get(groupId) || 0) > 1 ? 'concurrent' : 'contributing';
    }
  });
  const groups = new Map<string, any>();
  failures.filter((failure) => failure.classification !== 'terminal_summary').forEach((failure) => {
    const location = failure.location;
    const key = [failure.code, location.operation_id, location.tool_name, location.subagent_id].join(':');
    const group = groups.get(key) || { code: failure.code, location, event_ids: [], occurrence_count: 0, classifications: [] };
    group.event_ids.push(failure.event_id);
    group.occurrence_count += 1;
    if (!group.classifications.includes(failure.classification)) group.classifications.push(failure.classification);
    groups.set(key, group);
  });
  const source = terminal || primary;
  return {
    outcome: terminal?.kind === 'run.failed' ? 'failed' : terminal ? 'cancelled' : 'completed',
    summary: {
      code: source?.code || 'run_completed',
      message: source?.message || 'Run completed without a recorded failure.',
      retryable: Boolean(source?.retryable),
      primary_failure_event_id: primary?.event_id,
      primary_basis: primaryBasis,
      location: primary?.location || {},
      failure_count: failures.filter((failure) => failure.classification !== 'cancellation').length,
      cancellation_count: failures.filter((failure) => failure.classification === 'cancellation').length,
    },
    failures,
    groups: [...groups.values()],
    observability_gaps: terminal?.kind === 'run.failed' && nonTerminal.length === 0 ? [{ code: 'terminal_failure_without_lower_level_events', message: 'The runtime reported a terminal failure without lower-level diagnostic events.', terminal_event_id: terminal.event_id }] : [],
  };
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

export const buildCorrectiveInspection = (
  runDetails: AgentRunDetails,
  traceMetrics: Record<string, any> = {},
) => {
  const metrics = asObject(runDetails.metrics_json);
  const corrective = asObject(runDetails.corrective || metrics.corrective || traceMetrics.corrective);
  const retrievalQuality = asObject(runDetails.retrieval_quality_report || metrics.retrieval_quality_report || traceMetrics.retrieval_quality_report);
  const grounding = asObject(runDetails.grounding_report || metrics.grounding_report || metrics.grounding || traceMetrics.grounding_report || traceMetrics.grounding);
  if (!Object.keys(corrective).length && !Object.keys(retrievalQuality).length && !Object.keys(grounding).length) return undefined;
  return { corrective, retrievalQuality, grounding };
};

const retainedNodeStatus = (row: Record<string, any>): string | undefined => {
  const status = typeof row.status === 'string' ? row.status : undefined;
  const span = asObject(row.span);
  const spanEnded = Boolean(span.end_time ?? span.endTime);
  if (!spanEnded) return status;
  const normalized = normalizeAgentExecutionStatus(status);
  return normalized === 'active' || normalized === 'inactive' || normalized === 'planned'
    ? 'completed'
    : status;
};

const operationViewFromSummary = (row: Record<string, any>, nodeCatalog?: AgentNodeCatalog): TraceOperationView => {
  const raw = asObject(row.raw);
  const id = String(row.operation_id || row.id || raw.operation_id || 'unknown_operation');
  const type = typeof row.type === 'string'
    ? row.type
    : typeof row.operation_type === 'string'
      ? row.operation_type
      : undefined;
  return {
    id,
    type,
    label: asNonEmptyString(row.operation_label)
      || asNonEmptyString(row.label)
      || formatNodeLabel(id, type, nodeCatalog)
      || asNonEmptyString(row.node_name)
      || asNonEmptyString(raw.label)
      || asNonEmptyString(raw.node_name),
    instanceLabel: formatNodeInstanceLabel(id, type),
    parentOperationId: asNonEmptyString(row.parent_operation_id ?? raw.parent_operation_id),
    visitIndex: asNumber(row.visitIndex ?? row.visit_index ?? raw.visit_index ?? raw.visitIndex),
    status: retainedNodeStatus(row),
    skipped: row.skipped === true || row.status === 'skipped',
    durationMs: asNumber(row.durationMs ?? row.duration_ms),
    route: typeof row.route === 'string' ? row.route : undefined,
    routeReason: typeof row.routeReason === 'string' ? row.routeReason : typeof row.route_reason === 'string' ? row.route_reason : undefined,
    executionPlan: asStringArray(row.executionPlan ?? row.execution_plan),
    usedMemoryIdCount: asNumber(row.usedMemoryIdCount ?? row.used_memory_id_count ?? raw.used_memory_id_count),
    warningCodes: asStringArray(row.warningCodes ?? row.warnings),
    error: row.error && typeof row.error === 'object' ? row.error : undefined,
    span: row.span && typeof row.span === 'object' ? row.span : undefined,
    raw: { ...row, ...raw },
    topologyRef: asObject(row.topologyRef ?? row.topology_ref ?? raw.topology_ref),
  };
};

const toolViewFromSummary = (row: Record<string, any>): TraceToolView => {
  const raw = { ...row, ...asObject(row.payload), ...asObject(row.raw) };
  return {
    name: String(row.name || row.tool_name || raw.tool_name || 'tool'),
    id: typeof row.id === 'string' ? row.id : typeof row.tool_id === 'string' ? row.tool_id : typeof raw.tool_call_id === 'string' ? raw.tool_call_id : undefined,
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
    status: asNonEmptyString(row.status || raw.status),
    durationMs: asNumber(row.durationMs ?? row.elapsed_ms),
    sourceCount: asNumber(row.sourceCount ?? row.source_count),
    warningCodes: asStringArray(row.warningCodes ?? row.warnings),
    span: row.span && typeof row.span === 'object' ? row.span : undefined,
    raw,
  };
};

const modelViewFromSummary = (row: Record<string, any>): TraceModelView => {
  const raw = { ...row, ...asObject(row.payload), ...asObject(row.raw) };
  return {
    event_id: String(row.event_id || raw.event_id || `model:${row.invocation_id || 'unknown'}`),
    invocation_id: asNonEmptyString(row.invocation_id || raw.invocation_id),
    model_name: asNonEmptyString(row.model_name || raw.model_name),
    operation_id: asNonEmptyString(row.operation_id || raw.operation_id),
    operation_type: asNonEmptyString(row.operation_type || raw.operation_type),
    visit_index: asNumber(row.visit_index ?? raw.visit_index),
    subagent_id: asNonEmptyString(row.subagent_id || raw.subagent_id),
    parent_id: asNonEmptyString(row.parent_id || raw.parent_id),
    status: asNonEmptyString(row.status || raw.status),
    duration_ms: asNumber(row.duration_ms ?? raw.duration_ms),
    retry_count: asNumber(row.retry_count ?? raw.retry_count),
    response_chars: asNumber(row.response_chars ?? raw.response_chars),
    usage: asObject(row.usage || raw.usage) as Record<string, number>,
    error: row.error && typeof row.error === 'object' ? row.error : raw.error && typeof raw.error === 'object' ? raw.error : null,
    raw,
  };
};

const getRunGraph = (debug?: AgentRunDebug, nodeCatalog?: AgentNodeCatalog): TraceGraphView | undefined => {
  const graph = asObject(debug?.visualizations?.['langgraph.graph']);
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
      // Visualization descriptors describe topology; runtime overlays are optional.
      // Normalize the overlay fields here so a topology-only trace is still a
      // valid graph model for the generic canvas.
      toolSummaries: Array.isArray(node.toolSummaries) ? node.toolSummaries : [],
      rawEvents: Array.isArray(node.rawEvents) ? node.rawEvents : [],
      warningCount: asNumber(node.warningCount) ?? 0,
      errorCount: asNumber(node.errorCount) ?? 0,
      sourceCount: asNumber(node.sourceCount) ?? 0,
      artifactCount: asNumber(node.artifactCount) ?? 0,
      instanceId: id,
      instanceLabel: formatNodeInstanceLabel(id, type),
    };
  });
  const edges = asArray(graph.edges) as AgentGraphEdge[];
  if (nodes.length === 0 && edges.length === 0) return undefined;
  return {
    nodes,
    edges,
    executionPlan: asStringArray(graph.executionPlan ?? graph.execution_plan),
    selectedRoute: typeof (graph.selectedRoute ?? graph.selected_route) === 'string' ? String(graph.selectedRoute ?? graph.selected_route) : undefined,
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
    const summaryOperations = asArray(debug.operations).map((operation) => operationViewFromSummary(operation, options.nodeCatalog));
    const manifest = Array.isArray(debug.detail_manifest) ? debug.detail_manifest : [];
    const existingVisits = new Set(summaryOperations.map((operation) => `${operation.id}:${operation.visitIndex || 1}`));
    const manifestOperations = manifest
      .filter((detail) => !existingVisits.has(`${detail.operation_id}:${detail.visit_index || 1}`))
      .map((detail) => operationViewFromSummary({
        operation_id: detail.operation_id,
        operation_type: detail.operation_type,
        visitIndex: detail.visit_index,
        status: detail.status,
        raw: { detail_manifest: detail },
      }, options.nodeCatalog));
    const operations = [...summaryOperations, ...manifestOperations];
    const tools = asArray(debug.tools).map(toolViewFromSummary);
    const models = asArray(debug.models).map(modelViewFromSummary);
    const usedOperationCount = Math.max(asNumber(summary.usedOperationCount) ?? 0, operations.filter((operation) => !operation.skipped).length);
    const usedToolCount = asNumber(summary.usedToolCount) ?? tools.length;
    const memory = asObject(summary.memory);
    const diagnostics = debug.diagnostics as AgentTraceDiagnostics;
    return {
      debug,
      trace: getRunTrace(runDetails),
      graph: getRunGraph(debug, options.nodeCatalog),
      route: typeof summary.route === 'string' ? summary.route : typeof metrics.route === 'string' ? metrics.route : undefined,
      routeReason: typeof summary.routeReason === 'string' ? summary.routeReason : undefined,
      metrics,
      events: Array.isArray(debug.events) ? debug.events : [],
      visualizations: debug.visualizations || {},
      operations,
      tools,
      models,
      usedOperationCount,
      availableOperationCount: asNumber(summary.availableOperationCount),
      usedToolCount,
      availableToolCount: asNumber(summary.availableToolCount),
      warningCount: asNumber(summary.warningCount) ?? Number(metrics.tool_warning_count ?? 0),
      errorCount: diagnostics.summary.failure_count,
      diagnostics,
      parallelGroups: debug.parallel_groups,
      memory: Object.keys(memory).length > 0 ? {
        recalledMemoryIds: asStringArray(memory.recalledMemoryIds),
        searchedScopes: asArray(memory.searchedScopes),
        recalledCount: asNumber(memory.recalledCount) ?? 0,
      } : undefined,
      finalOutput: runDetails.final_output || debug.final_output,
      detailManifest: manifest,
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
  const operations: TraceOperationView[] = [];
  const operationIndex = new Map<string, number>();
  const tools: TraceToolView[] = [];
  const toolIndex = new Map<string, number>();
  const models: TraceModelView[] = [];
  let finalOutput: AgentRunFinalOutput | undefined;
  let route: string | undefined;
  let routeReason: string | undefined;
  const timelineEvents: AgentTraceTimelineEvent[] = events.map((envelope, index) => ({
    event_id: String((envelope.data as any)?.event_id || `live:${index + 1}`),
    sequence: Number((envelope.data as any)?.sequence || index + 1),
    kind: envelope.event,
    occurred_at: (envelope.data as any)?.occurred_at,
    operation_id: (envelope.data as any)?.operation_id,
    parallel_group_id: (envelope.data as any)?.parallel_group_id ?? (envelope.data as any)?.dispatch_id ?? ((envelope.data as any)?.wave_id !== undefined ? String((envelope.data as any).wave_id) : undefined),
    parallel_member_id: (envelope.data as any)?.work_id,
    parallel_attempt: asNumber((envelope.data as any)?.attempt),
    payload: Object.fromEntries(Object.entries(asObject(envelope.data)).filter(([key]) => !['response', 'runtime_binding', 'runtime_metadata', 'prompt', 'messages', 'headers', 'arguments', 'args', 'framework_details', 'framework_metadata', 'parallel_groups'].includes(key))),
    framework_details: asObject((envelope.data as any)?.framework_details),
  }));
  const latestParallelSnapshot = [...events].reverse().find((envelope) => Array.isArray((envelope.data as any)?.parallel_groups));
  const liveParallelGroups = latestParallelSnapshot ? (latestParallelSnapshot.data as any).parallel_groups : [];
  // A live snapshot may be projected from the runtime's complete canonical
  // journal while this client still has only a suffix of that journal (for
  // example after reconnecting with an after-sequence cursor). Structural
  // validation remains strict, but referential validation must wait for the
  // retained/full trace, where the complete event set is available.
  const parallelError = validateParallelGroups(liveParallelGroups, timelineEvents, { requireEventReferences: false });
  if (parallelError) {
    return {
      parseError: parallelError.slice(0, 240),
      parseCorrelationId: 'trace:live',
      metrics: {},
      events: [],
      visualizations: {},
      operations: [],
      tools: [],
      models: [],
      usedOperationCount: 0,
      usedToolCount: 0,
      warningCount: 0,
      errorCount: 0,
      diagnostics: buildDiagnosticsFromTimeline([]),
      parallelGroups: [],
      detailManifest: [],
    };
  }

  events.forEach((envelope, index) => {
    const data = asObject(envelope.data);
    if (envelope.event.startsWith('operation.') && typeof data.operation_id === 'string') {
      const operationId = String(data.operation_id);
      const visitIndex = asNumber(data.visit_index) || 1;
      const key = `${operationId}:${visitIndex}`;
      const status = envelope.event.endsWith('.started') ? 'active'
        : envelope.event.endsWith('.failed') ? 'error'
          : envelope.event.endsWith('.skipped') ? 'skipped'
            : 'completed';
      const rawError = data.detail?.error ?? data.error;
      const row: TraceOperationView = {
        id: operationId,
        type: asNonEmptyString(data.operation_type),
        label: asNonEmptyString(data.operation_label) || operationId,
        instanceLabel: formatNodeInstanceLabel(operationId, asNonEmptyString(data.operation_type)),
        parentOperationId: asNonEmptyString(data.parent_operation_id),
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
        topologyRef: asObject(data.topology_ref),
      };
      const existing = operationIndex.get(key);
      if (existing === undefined) {
        operationIndex.set(key, operations.length);
        operations.push(row);
      } else {
        operations[existing] = { ...operations[existing], ...row, raw: { ...operations[existing].raw, ...data } };
      }
      route = row.route || route;
      routeReason = row.routeReason || routeReason;
    }
    if (envelope.event.startsWith('tool.')) {
      const tool = toolViewFromSummary({ ...data, status: data.status || envelope.event.slice(5), ok: envelope.event !== 'tool.failed' && data.ok !== false });
      const key = String(tool.id || `${tool.name}:${tool.callerNode || ''}:${tool.callerVisitIndex || 1}`);
      const existing = toolIndex.get(key);
      if (existing === undefined) {
        toolIndex.set(key, tools.length);
        tools.push(tool);
      } else {
        tools[existing] = { ...tools[existing], ...tool, raw: { ...tools[existing].raw, ...tool.raw } };
      }
    }
    if (envelope.event.startsWith('llm.')) {
      const model = modelViewFromSummary({ ...data, event_id: data.event_id || `live:${index + 1}`, status: envelope.event.slice(4) });
      const existing = models.findIndex((row) => row.invocation_id === model.invocation_id);
      if (existing >= 0) models[existing] = { ...models[existing], ...model, raw: { ...models[existing].raw, ...model.raw } };
      else models.push(model);
    }
    if (envelope.event === 'run.completed') {
      finalOutput = asObject(data.final_output) as AgentRunFinalOutput;
      if (!finalOutput.answer && typeof data.answer === 'string') finalOutput.answer = data.answer;
      route = asNonEmptyString(data.route) || route;
      routeReason = asNonEmptyString(data.route_reason) || routeReason;
    }
  });

  const diagnostics = buildDiagnosticsFromTimeline(timelineEvents);
  const visualizations: Record<string, AgentTraceVisualization> = {
    'generic.timeline': { id: 'generic.timeline' },
  };
  if (liveParallelGroups.length > 0) {
    visualizations['generic.parallel'] = {
      id: 'generic.parallel',
      group_ids: liveParallelGroups.map((group: AgentTraceParallelGroup) => group.group_id),
    };
  }

  return {
    route,
    routeReason,
    metrics: {},
    events: timelineEvents,
    visualizations,
    operations,
    tools,
    models,
    usedOperationCount: new Set(operations.filter((operation) => !operation.skipped).map((operation) => operation.id)).size,
    usedToolCount: tools.length,
    warningCount: operations.reduce((count, operation) => count + operation.warningCodes.length, 0) + tools.reduce((count, tool) => count + tool.warningCodes.length, 0),
    errorCount: diagnostics.summary.failure_count,
    diagnostics,
    parallelGroups: liveParallelGroups as AgentTraceParallelGroup[],
    finalOutput,
    detailManifest: operations.filter((operation) => operation.raw.detail).map((operation) => ({
      operation_id: operation.id,
      operation_type: operation.type,
      visit_index: operation.visitIndex || 1,
      status: operation.status,
      available: true,
      truncated: Boolean(operation.raw.detail?.safety?.truncated),
    })),
  };
};

export const mergeLiveAndRetainedTraceViews = (
  live: TraceRunView,
  retained?: TraceRunView,
): TraceRunView => {
  if (!retained) return live;
  const liveVisits = new Set(live.operations.map((operation) => `${operation.id}:${operation.visitIndex || 1}`));
  const operations = [
    ...retained.operations.filter((operation) => !liveVisits.has(`${operation.id}:${operation.visitIndex || 1}`)),
    ...live.operations,
  ];
  const toolKeys = new Set(live.tools.map((tool, index) => `${tool.id || tool.name}:${tool.callerNode || ''}:${tool.callerVisitIndex || 1}:${index}`));
  const retainedTools = retained.tools.filter((tool, index) => !toolKeys.has(`${tool.id || tool.name}:${tool.callerNode || ''}:${tool.callerVisitIndex || 1}:${index}`));
  const tools = [...retainedTools, ...live.tools];
  const modelKeys = new Set(live.models.map((model) => model.invocation_id || model.event_id));
  const retainedModels = retained.models.filter((model) => !modelKeys.has(model.invocation_id || model.event_id));
  const models = [...retainedModels, ...live.models];
  const detailManifest = new Map(
    [...retained.detailManifest, ...live.detailManifest]
      .map((row) => [`${row.operation_id}:${row.visit_index}`, row] as const),
  );
  return {
    ...retained,
    ...live,
    graph: retained.graph || live.graph,
    events: live.events.length > 0 ? live.events : retained.events,
    visualizations: { ...retained.visualizations, ...live.visualizations },
    route: live.route || retained.route,
    routeReason: live.routeReason || retained.routeReason,
    operations,
    tools,
    models,
    usedOperationCount: new Set(operations.filter((operation) => !operation.skipped).map((operation) => operation.id)).size,
    usedToolCount: tools.length,
    warningCount: operations.reduce((count, operation) => count + operation.warningCodes.length, 0)
      + tools.reduce((count, tool) => count + tool.warningCodes.length, 0),
    errorCount: live.diagnostics.summary.failure_count || retained.diagnostics.summary.failure_count,
    diagnostics: live.events.length > 0 ? live.diagnostics : retained.diagnostics,
    parallelGroups: live.parallelGroups.length > 0 ? live.parallelGroups : retained.parallelGroups,
    finalOutput: live.finalOutput || retained.finalOutput,
    detailManifest: [...detailManifest.values()],
  };
};

export const buildTraceExportJson = (view?: TraceRunView): string => {
  if (!view) return '';
  return JSON.stringify({
    ...(view.debug || {}),
    diagnostics: view.diagnostics,
    events: view.events,
    operations: view.operations,
    tools: view.tools,
    models: view.models,
    parallel_groups: view.parallelGroups,
    visualizations: view.visualizations,
    final_output: view.finalOutput,
  }, null, 2);
};
