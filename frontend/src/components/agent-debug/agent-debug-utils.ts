import type { AgentRunDetails } from '../../lib/api';

const asObject = (value: any): Record<string, any> => (
  value && typeof value === 'object' && !Array.isArray(value) ? value : {}
);

const asArray = (value: any): Record<string, any>[] => (
  Array.isArray(value) ? value.filter((item): item is Record<string, any> => item && typeof item === 'object') : []
);

export const getNodeEventName = (event: Record<string, any>) => String(event?.node || event?.name || 'unknown_node');

export const getToolEventName = (event: Record<string, any>) => String(
  event?.tool_display_name || event?.tool_name || event?.tool_id || 'unknown_tool'
);

export const isSkippedNodeEvent = (event: Record<string, any>) => event?.status === 'skipped' || event?.skipped === true;

export const formatTraceError = (error: unknown) => {
  if (!error) return null;
  if (typeof error === 'string') return error;
  if (typeof error === 'object') {
    const err = error as Record<string, any>;
    return String(err.message || err.code || err.type || JSON.stringify(err));
  }
  return String(error);
};

export const getRunTrace = (runDetails: AgentRunDetails): Record<string, any> | undefined => {
  const trace = runDetails.debug?.trace;
  return trace && typeof trace === 'object' && !Array.isArray(trace) ? trace : undefined;
};

export const getRunDebugMetrics = (runDetails: AgentRunDetails) => {
  const trace = getRunTrace(runDetails);
  if (trace?.metrics && typeof trace.metrics === 'object') return trace.metrics;
  const debug = runDetails.debug;
  return debug?.metrics || runDetails.metrics_json || {};
};

const promptSummaryFromSpan = (span: Record<string, any>) => {
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

const eventFromNodeSpan = (span: Record<string, any>): Record<string, any> => {
  const raw = asObject(span.raw);
  if (raw.node || raw.name) return raw;
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
  };
};

const eventFromToolSpan = (span: Record<string, any>): Record<string, any> => {
  const raw = asObject(span.raw);
  if (raw.tool_name) return raw;
  const attributes = asObject(span.attributes);
  const input = asObject(span.input);
  const output = asObject(span.output);
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
    artifact_keys: attributes['askpdf.artifact_keys'],
    known_warning_codes: attributes['askpdf.known_warning_codes'],
    tool_input: input.value,
    result_preview: output.value,
    artifact_refs: output.refs,
    artifact_summary: output.summary,
  };
};

export const getRunNodeEvents = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const trace = getRunTrace(runDetails);
  if (trace) {
    const spans = asArray(trace.spans);
    const events = spans
      .filter((span) => asObject(span.attributes)['askpdf.node.id'])
      .map(eventFromNodeSpan);
    if (events.length > 0) return events;
  }
  const events = runDetails.debug?.node_events;
  return asArray(events);
};

export const getRunToolEvents = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const trace = getRunTrace(runDetails);
  if (trace) {
    const spans = asArray(trace.spans);
    const events = spans
      .filter((span) => span.kind === 'TOOL' || asObject(span.attributes)['tool.name'])
      .map(eventFromToolSpan);
    if (events.length > 0) return events;
  }
  const events = runDetails.debug?.tool_events;
  return asArray(events);
};

export const getAvailableNodeCount = (runDetails: AgentRunDetails) => {
  const nodes = runDetails.resolved_spec_json?.config?.graph?.nodes;
  return Array.isArray(nodes) ? nodes.length : undefined;
};

export const getAvailableToolCount = (runDetails: AgentRunDetails) => {
  const toolIds = runDetails.resolved_spec_json?.config?.allowed_tool_ids;
  return Array.isArray(toolIds) ? new Set(toolIds.filter(Boolean)).size : undefined;
};
