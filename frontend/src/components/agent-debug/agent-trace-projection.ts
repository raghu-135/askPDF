import type { AgentDebugTrace, AgentRunDetails, AgentTraceSpan } from '../../lib/api';
import type { AgentGraphRuntimeOverlay } from '../agent-graph/agent-graph-types';

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
  span?: AgentTraceSpan;
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
  span?: AgentTraceSpan;
  raw: Record<string, any>;
}

export interface TraceRunView {
  trace?: AgentDebugTrace;
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

const isSkippedNodeEvent = (event: Record<string, any>) => (
  event?.skipped === true || event?.status === 'skipped'
);

const asStringArray = (value: any): string[] => (
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
);

export const getRunTrace = (runDetails: AgentRunDetails): AgentDebugTrace | undefined => {
  const trace = runDetails.debug?.trace;
  return trace && typeof trace === 'object' && !Array.isArray(trace) ? trace : undefined;
};

export const getRunDebugMetrics = (runDetails: AgentRunDetails) => {
  const trace = getRunTrace(runDetails);
  if (trace?.metrics && typeof trace.metrics === 'object') return trace.metrics;
  const debug = runDetails.debug;
  return debug?.metrics || runDetails.metrics_json || {};
};

const promptSummaryFromSpan = (span: AgentTraceSpan) => {
  const promptEvent = asArray(span.events).find((event) => event.name === 'prompt.rendered');
  if (!promptEvent) return undefined;
  const attributes = asObject(promptEvent.attributes);
  const output = asObject(promptEvent.output);
  return {
    section: attributes['prompt.name'],
    prompt_chars: attributes['prompt.chars'],
    system_message: output.system_message,
    preview: output.preview,
  };
};

const llmResultFromSpan = (span: AgentTraceSpan) => {
  const decision = asArray(span.events).find((event) => event.name === 'decision.made');
  if (!decision) return undefined;
  const attributes = asObject(decision.attributes);
  return {
    route: attributes['askpdf.route'],
    route_reason: attributes['askpdf.route_reason'],
    execution_plan: attributes['askpdf.execution_plan'],
  };
};

const eventFromNodeSpan = (span: AgentTraceSpan): Record<string, any> => {
  const raw = asObject(span.raw);
  if (raw.node || raw.name) {
    return {
      ...raw,
      __trace_span: span,
      __trace_span_id: span.span_id,
      __trace_kind: span.kind,
    };
  }
  const attributes = asObject(span.attributes);
  const input = asObject(span.input);
  const output = asObject(span.output);
  const nodeId = String(attributes['askpdf.node.id'] || span.name || 'unknown_node');
  return {
    node: nodeId,
    status: span.status,
    skipped: span.status === 'skipped',
    skip_reason: attributes['askpdf.skip_reason'],
    elapsed_ms: span.duration_ms,
    route: attributes['askpdf.route'],
    route_reason: attributes['askpdf.route_reason'],
    execution_plan: attributes['askpdf.execution_plan'],
    evidence_chars: attributes['askpdf.evidence_chars'],
    answer_chars: attributes['askpdf.answer_chars'],
    input_refs: input.refs,
    input_preview: input.value,
    output_refs: output.refs,
    output_preview: output.value,
    prompt_summary: promptSummaryFromSpan(span),
    llm_result_summary: llmResultFromSpan(span),
    warnings: asArray(span.events)
      .filter((event) => event.name === 'warning')
      .map((event) => asObject(event.attributes)['warning.code'])
      .filter(Boolean),
    error: asArray(span.events).find((event) => event.name === 'exception')?.attributes,
    __trace_span: span,
    __trace_span_id: span.span_id,
    __trace_kind: span.kind,
  };
};

const eventFromToolSpan = (span: AgentTraceSpan): Record<string, any> => {
  const raw = asObject(span.raw);
  if (raw.tool_name) {
    return {
      ...raw,
      __trace_span: span,
      __trace_span_id: span.span_id,
      __trace_kind: span.kind,
    };
  }
  const attributes = asObject(span.attributes);
  const input = asObject(span.input);
  const output = asObject(span.output);
  const warningEvents = asArray(span.events).filter((event) => event.name === 'warning');
  return {
    tool_name: attributes['tool.name'] || span.name,
    tool_id: attributes['tool.id'],
    tool_category: attributes['askpdf.tool.category'],
    tool_display_name: attributes['tool.description'] || span.name,
    caller_node: attributes['askpdf.caller_node'],
    ok: span.status !== 'error',
    elapsed_ms: span.duration_ms,
    result_chars: attributes['askpdf.result_chars'],
    source_count: attributes['askpdf.source_count'],
    warnings: warningEvents.map((event) => asObject(event.attributes)['warning.code']).filter(Boolean),
    error: asArray(span.events).find((event) => event.name === 'exception')?.attributes,
    artifact_keys: attributes['askpdf.artifact_keys'],
    known_warning_codes: attributes['askpdf.known_warning_codes'],
    tool_input: input.value,
    result_preview: output.value,
    artifact_refs: output.refs,
    artifact_summary: output.summary,
    __trace_span: span,
    __trace_span_id: span.span_id,
    __trace_kind: span.kind,
  };
};

const getTraceNodeSpans = (trace?: AgentDebugTrace): AgentTraceSpan[] => (
  (trace?.spans || []).filter((span) => Boolean(asObject(span.attributes)['askpdf.node.id']))
);

const getTraceToolSpans = (trace?: AgentDebugTrace): AgentTraceSpan[] => (
  (trace?.spans || []).filter((span) => span.kind === 'TOOL' || Boolean(asObject(span.attributes)['tool.name']))
);

const getRunNodeEventRows = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const trace = getRunTrace(runDetails);
  const traceEvents = getTraceNodeSpans(trace).map(eventFromNodeSpan);
  if (traceEvents.length > 0) return traceEvents;
  return asArray(runDetails.debug?.node_events);
};

const getRunToolEventRows = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const trace = getRunTrace(runDetails);
  const traceEvents = getTraceToolSpans(trace).map(eventFromToolSpan);
  if (traceEvents.length > 0) return traceEvents;
  return asArray(runDetails.debug?.tool_events);
};

const nodeViewFromRow = (event: Record<string, any>): TraceNodeView => ({
  id: String(event.node || event.name || 'unknown_node'),
  status: typeof event.status === 'string' ? event.status : undefined,
  skipped: isSkippedNodeEvent(event),
  durationMs: Number.isFinite(Number(event.elapsed_ms)) ? Number(event.elapsed_ms) : undefined,
  route: typeof event.route === 'string' ? event.route : undefined,
  routeReason: typeof event.route_reason === 'string' ? event.route_reason : undefined,
  executionPlan: asStringArray(event.execution_plan),
  warningCodes: asStringArray(event.warnings),
  error: event.error && typeof event.error === 'object' ? event.error : undefined,
  span: event.__trace_span && typeof event.__trace_span === 'object' ? event.__trace_span : undefined,
  raw: event,
});

const toolViewFromRow = (event: Record<string, any>): TraceToolView => ({
  name: String(event.tool_name || 'tool'),
  id: typeof event.tool_id === 'string' ? event.tool_id : undefined,
  category: typeof event.tool_category === 'string' ? event.tool_category : undefined,
  displayName: typeof event.tool_display_name === 'string' ? event.tool_display_name : undefined,
  callerNode: typeof event.caller_node === 'string' ? event.caller_node : undefined,
  ok: event.ok !== false,
  durationMs: Number.isFinite(Number(event.elapsed_ms)) ? Number(event.elapsed_ms) : undefined,
  sourceCount: Number.isFinite(Number(event.source_count)) ? Number(event.source_count) : undefined,
  warningCodes: asStringArray(event.warnings),
  span: event.__trace_span && typeof event.__trace_span === 'object' ? event.__trace_span : undefined,
  raw: event,
});

export const buildRunTraceView = (runDetails: AgentRunDetails): TraceRunView => {
  const trace = getRunTrace(runDetails);
  const metrics = getRunDebugMetrics(runDetails);
  const attributes = asObject(trace?.attributes);
  const debug = runDetails.debug;
  const nodes = getRunNodeEventRows(runDetails).map(nodeViewFromRow);
  const tools = getRunToolEventRows(runDetails).map(toolViewFromRow);
  const availableNodeCount = Array.isArray(runDetails.resolved_spec_json?.config?.graph?.nodes)
    ? runDetails.resolved_spec_json.config.graph.nodes.length
    : undefined;
  const availableToolCount = Array.isArray(runDetails.resolved_spec_json?.config?.allowed_tool_ids)
    ? new Set(runDetails.resolved_spec_json.config.allowed_tool_ids.filter(Boolean)).size
    : undefined;
  const usedNodeCount = nodes.filter((node) => !node.skipped).length;
  const usedToolCount = Number(metrics.tool_event_count ?? runDetails.debug?.tool_event_count ?? tools.length ?? 0);
  const warningCount = Number(metrics.tool_warning_count ?? runDetails.debug?.tool_warning_count ?? 0);
  const errorCount = Number(metrics.error_count ?? runDetails.debug?.error_count ?? metrics.tool_error_count ?? runDetails.debug?.tool_error_count ?? 0);
  const error = debug?.error;

  return {
    trace,
    route: attributes['askpdf.route'] || debug?.route || metrics.route,
    routeReason: attributes['askpdf.route_reason'] || debug?.route_reason,
    metrics,
    nodes,
    tools,
    usedNodeCount,
    availableNodeCount,
    usedToolCount,
    availableToolCount,
    warningCount: Number.isFinite(warningCount) ? warningCount : 0,
    errorCount: Number.isFinite(errorCount) ? errorCount : 0,
    errors: error && typeof error === 'object' ? [error] : [],
  };
};

export const buildRunGraphOverlay = (view: TraceRunView): AgentGraphRuntimeOverlay => ({
  route: view.route,
  routeReason: view.routeReason,
  nodeEvents: view.nodes.map((node) => node.raw),
  toolEvents: view.tools.map((tool) => tool.raw),
  errors: view.errors,
  metrics: view.metrics,
});
