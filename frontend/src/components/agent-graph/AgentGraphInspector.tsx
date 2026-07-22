import React, { useState } from 'react';
import { Box, Button, Chip, Tab, Tabs, Typography } from '@mui/material';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import type { TraceNodeView } from '../agent-debug/agent-trace-projection';
import type { AgentGraphSelection, AgentNodeVisitRef } from './agent-graph-types';
import { agentNodeVisitKey, getNodeVisitRoute, normalizeVisitIndex } from './agent-node-visits';
import {
  DetailLine,
  hasValue,
  InspectorSection,
  JsonPreview,
} from './AgentGraphInspectorPrimitives';

const sectionBg = 'rgba(0,0,0,0.03)';

interface AgentGraphInspectorProps {
  selection: AgentGraphSelection;
  selectedVisitRef?: AgentNodeVisitRef | null;
  visits?: TraceNodeView[];
  onSelectVisit?: (visit: AgentNodeVisitRef) => void;
  onViewVisit?: (visit: AgentNodeVisitRef) => void;
}

const withoutInternalTraceFields = (value: Record<string, any>) => {
  const { __trace_span, ...rest } = value;
  return rest;
};

export default function AgentGraphInspector({
  selection,
  selectedVisitRef,
  visits = [],
  onSelectVisit,
  onViewVisit,
}: AgentGraphInspectorProps) {
  const [tab, setTab] = useState<'details' | 'raw'>('details');

  if (!selection) {
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: sectionBg }}>
        <Typography variant="caption" color="text.secondary">
          Select a node or edge for details.
        </Typography>
      </Box>
    );
  }

  if (selection.kind === 'edge') {
    const { edge } = selection;
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: sectionBg }}>
        <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
          {edge.conditional ? 'Route Edge' : 'Sequential Edge'}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
          {edge.label && <Chip size="small" label={edge.label} color={edge.selected ? 'primary' : 'default'} variant={edge.selected ? 'filled' : 'outlined'} />}
          <Chip size="small" label={edge.conditional ? 'conditional' : 'sequential'} variant="outlined" />
          <Chip size="small" label={edge.active ? 'active' : 'inactive'} variant="outlined" />
          {edge.selected && <Chip size="small" color="primary" label="selected" />}
        </Box>
        <DetailLine label="From" value={edge.source} />
        <DetailLine label="To" value={edge.target} />
        {edge.route && <DetailLine label="Route" value={edge.route} />}
        {edge.raw && <JsonPreview value={edge.raw} />}
      </Box>
    );
  }

  const { node } = selection;
  const nodeElapsed = formatDurationMs(node.elapsedMs);
  const skipReason = formatSkipReason(node.skipReason);
  const statusLabel = node.status === 'skipped' ? skipReason || 'Skipped' : node.status;
  const instanceLabel = node.instanceLabel || node.id;
  const observability = node.observability && typeof node.observability === 'object'
    ? node.observability as Record<string, unknown>
    : {};
  const capabilities = Array.isArray(node.capabilities)
    ? node.capabilities.filter((item): item is string => typeof item === 'string' && item.length > 0)
    : [];
  const nodeVisits = visits
    .map((visit) => ({ visit, visitIndex: normalizeVisitIndex(visit.visitIndex) }))
    .filter(({ visit }) => visit.id === node.id || visit.id === node.instanceId);
  const selectedPosition = selectedVisitRef && (selectedVisitRef.nodeId === node.id || selectedVisitRef.nodeId === node.instanceId)
    ? nodeVisits.findIndex(({ visit }) => agentNodeVisitKey(visit) === agentNodeVisitKey(selectedVisitRef))
    : -1;
  const effectivePosition = selectedPosition >= 0 ? selectedPosition : nodeVisits.length - 1;
  const selectedVisit = effectivePosition >= 0 ? nodeVisits[effectivePosition] : undefined;
  const selectedGraphVisit = selectedVisit
    ? node.visits?.find((visit) => normalizeVisitIndex(visit.visitIndex) === selectedVisit.visitIndex)
    : undefined;
  const selectedVisitRefValue = selectedVisit
    ? { nodeId: selectedVisit.visit.id, visitIndex: selectedVisit.visitIndex }
    : undefined;
  const selectedVisitElapsed = formatDurationMs(selectedVisit?.visit.durationMs ?? selectedGraphVisit?.elapsedMs);
  const selectedVisitRoute = getNodeVisitRoute(selectedVisit?.visit) || selectedGraphVisit?.evaluatorRoute || selectedGraphVisit?.route;
  const selectedWarningCount = Math.max(selectedVisit?.visit.warningCodes.length || 0, selectedGraphVisit?.warningCount || 0);
  const selectedErrorCount = Math.max(selectedVisit?.visit.error ? 1 : 0, selectedGraphVisit?.errorCount || 0);
  const toolTraceSpans = node.toolSummaries
    .map((tool) => tool.traceSpan)
    .filter((span): span is Record<string, any> => Boolean(span));
  const rawPayload = {
    focused_span_ids: node.focusedSpanIds || [],
    focused_trace_spans: node.focusedTraceSpans || [],
    trace_spans: node.traceSpans || [],
    tool_trace_spans: toolTraceSpans,
    visits: node.visits || [],
    node_rows: node.rawEvents.map(withoutInternalTraceFields),
    tool_rows: node.toolSummaries.map((tool) => withoutInternalTraceFields(tool.raw)),
  };

  const selectPosition = (position: number) => {
    const target = nodeVisits[position];
    if (target) onSelectVisit?.({ nodeId: target.visit.id, visitIndex: target.visitIndex });
  };

  return (
    <Box sx={{ p: 1, borderRadius: 1, bgcolor: sectionBg }}>
      <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
        Node: {node.label}
      </Typography>
      {instanceLabel && (
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: 0.25 }}>
          {instanceLabel}
        </Typography>
      )}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
        <Chip size="small" label={statusLabel} variant="outlined" />
        {nodeElapsed && <Chip size="small" label={nodeElapsed} variant="outlined" />}
        {nodeVisits.length > 1 && <Chip size="small" label={`${nodeVisits.length} visits`} variant="outlined" />}
        {node.category && <Chip size="small" label={node.category} variant="outlined" />}
        {node.sourceCount > 0 && <Chip size="small" label={`${node.sourceCount} sources`} variant="outlined" />}
        {node.artifactCount > 0 && <Chip size="small" label={`${node.artifactCount} artifacts`} variant="outlined" />}
        {node.warningCount > 0 && <Chip size="small" color="warning" label={`${node.warningCount} warnings`} />}
        {node.errorCount > 0 && <Chip size="small" color="error" label={`${node.errorCount} errors`} />}
        {node.focused && <Chip size="small" color="primary" label="focused" />}
      </Box>

      <Tabs
        value={tab}
        onChange={(_, value) => setTab(value)}
        sx={{ mt: 1, minHeight: 32, '& .MuiTab-root': { minHeight: 32, py: 0, fontSize: '0.72rem' } }}
      >
        <Tab value="details" label="Details" />
        <Tab value="raw" label="Raw JSON" />
      </Tabs>

      {tab === 'details' ? (
        <>
          {selectedVisit && selectedVisitRefValue && (
            <InspectorSection title="Selected invocation">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="caption" sx={{ fontWeight: 700 }}>
                  Visit {effectivePosition + 1} of {nodeVisits.length}
                </Typography>
                {nodeVisits.length > 1 && (
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Button aria-label={`Previous invocation of ${node.label}`} size="small" variant="outlined" disabled={effectivePosition <= 0} onClick={() => selectPosition(effectivePosition - 1)}>
                      Previous
                    </Button>
                    <Button aria-label={`Next invocation of ${node.label}`} size="small" variant="outlined" disabled={effectivePosition >= nodeVisits.length - 1} onClick={() => selectPosition(effectivePosition + 1)}>
                      Next
                    </Button>
                  </Box>
                )}
              </Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
                {selectedVisit.visit.status && <Chip size="small" label={selectedVisit.visit.status} variant="outlined" />}
                {selectedVisitElapsed && <Chip size="small" label={selectedVisitElapsed} variant="outlined" />}
                {selectedVisitRoute && <Chip size="small" label={`route ${selectedVisitRoute}`} variant="outlined" />}
                {Boolean(selectedGraphVisit?.toolCount) && <Chip size="small" label={`${selectedGraphVisit?.toolCount} tools`} variant="outlined" />}
                {selectedWarningCount > 0 && <Chip size="small" color="warning" label={`${selectedWarningCount} warnings`} />}
                {selectedErrorCount > 0 && <Chip size="small" color="error" label={`${selectedErrorCount} errors`} />}
              </Box>
              {(selectedVisit.visit.routeReason || selectedGraphVisit?.routeReason) && (
                <DetailLine label="Route reason" value={selectedVisit.visit.routeReason || selectedGraphVisit?.routeReason} />
              )}
              <Button aria-label={`View ${node.label} invocation details in Execution Progress`} size="small" sx={{ mt: 0.75 }} onClick={() => onViewVisit?.(selectedVisitRefValue)}>
                View details in Execution Progress
              </Button>
            </InspectorSection>
          )}

          <DetailLine label="Instance" value={node.id} />
          <DetailLine label="Type" value={node.type} />
          {(node.category || capabilities.length > 0 || hasValue(observability)) && (
            <InspectorSection title="Metadata">
              <DetailLine label="Category" value={node.category} />
              <DetailLine label="Capabilities" value={capabilities.length ? capabilities.join(', ') : undefined} />
              <DetailLine label="Span kind" value={typeof observability.span_kind === 'string' ? observability.span_kind : undefined} />
              <DetailLine label="Event prefix" value={typeof observability.event_prefix === 'string' ? observability.event_prefix : undefined} />
            </InspectorSection>
          )}
          {(node.route || node.routeReason || node.executionPlan?.length) && (
            <InspectorSection title="Decision summary">
              <DetailLine label="Latest route" value={node.route} />
              <DetailLine label="Latest reason" value={node.routeReason} />
              <DetailLine label="Execution plan" value={node.executionPlan?.length ? node.executionPlan.join(' -> ') : undefined} />
            </InspectorSection>
          )}
          {node.toolSummaries.length > 0 && (
            <InspectorSection title="Tool summary">
              {node.toolSummaries.map((tool, index) => (
                <Typography key={`${tool.toolName}-${index}`} variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: index ? 0.5 : 0 }}>
                  {tool.displayName || tool.toolName}: {tool.ok ? 'ok' : 'failed'}
                  {formatDurationMs(tool.elapsedMs) ? `, ${formatDurationMs(tool.elapsedMs)}` : ''}
                </Typography>
              ))}
            </InspectorSection>
          )}
          {(node.warnings?.length || node.toolSummaries.some((tool) => tool.warnings.length > 0)) && (
            <InspectorSection title="Warnings">
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {(node.warnings || []).map((warning, index) => (
                  <Chip key={`node-warning-${index}`} size="small" color="warning" label={warning} />
                ))}
                {node.toolSummaries.flatMap((tool) => tool.warnings).map((warning, index) => (
                  <Chip key={`tool-warning-${index}`} size="small" color="warning" label={warning} variant="outlined" />
                ))}
              </Box>
            </InspectorSection>
          )}
        </>
      ) : (
        <InspectorSection title="Raw graph data">
          <JsonPreview value={rawPayload} maxHeight={360} />
        </InspectorSection>
      )}
    </Box>
  );
}
