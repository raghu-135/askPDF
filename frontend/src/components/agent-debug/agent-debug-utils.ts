import type { AgentRunDetails } from '../../lib/api';

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

export const getRunDebugMetrics = (runDetails: AgentRunDetails) => {
  const debug = runDetails.debug;
  return debug?.metrics || runDetails.metrics_json || {};
};

export const getRunNodeEvents = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const events = runDetails.debug?.node_events;
  return Array.isArray(events) ? events : [];
};

export const getRunToolEvents = (runDetails: AgentRunDetails): Record<string, any>[] => {
  const events = runDetails.debug?.tool_events;
  return Array.isArray(events) ? events : [];
};

export const getAvailableNodeCount = (runDetails: AgentRunDetails) => {
  const nodes = runDetails.resolved_spec_json?.config?.graph?.nodes;
  return Array.isArray(nodes) ? nodes.length : undefined;
};

export const getAvailableToolCount = (runDetails: AgentRunDetails) => {
  const toolIds = runDetails.resolved_spec_json?.config?.allowed_tool_ids;
  return Array.isArray(toolIds) ? new Set(toolIds.filter(Boolean)).size : undefined;
};
