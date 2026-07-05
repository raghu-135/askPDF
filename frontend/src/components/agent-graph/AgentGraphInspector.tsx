import React, { useState } from 'react';
import { Box, Chip, Tab, Tabs, Typography } from '@mui/material';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentGraphSelection } from './agent-graph-types';
import {
  DetailLine,
  hasValue,
  InspectorSection,
  JsonPreview,
  TraceObject,
} from './AgentGraphInspectorPrimitives';

const sectionBg = 'rgba(0,0,0,0.03)';

const withoutInternalTraceFields = (value: Record<string, any>) => {
  const { __trace_span, ...rest } = value;
  return rest;
};

export default function AgentGraphInspector({ selection }: { selection: AgentGraphSelection }) {
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
  const toolTraceSpans = node.toolSummaries
    .map((tool) => tool.traceSpan)
    .filter((span): span is Record<string, any> => Boolean(span));
  const hasTracePayload = Boolean(node.traceSpans?.length || toolTraceSpans.length);
  const rawPayload = hasTracePayload
    ? {
      trace_spans: node.traceSpans || [],
      tool_trace_spans: toolTraceSpans,
      graph_node_rows: node.rawEvents.map(withoutInternalTraceFields),
      graph_tool_rows: node.toolSummaries.map((tool) => withoutInternalTraceFields(tool.raw)),
    }
    : {
      node_events: node.rawEvents,
      tool_events: node.toolSummaries.map((tool) => tool.raw),
    };
  return (
    <Box sx={{ p: 1, borderRadius: 1, bgcolor: sectionBg }}>
      <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
        Node: {node.label}
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
        <Chip size="small" label={statusLabel} variant="outlined" />
        {nodeElapsed && <Chip size="small" label={nodeElapsed} variant="outlined" />}
        {node.route && <Chip size="small" label={`route ${node.route}`} variant="outlined" />}
        {node.sourceCount > 0 && <Chip size="small" label={`${node.sourceCount} sources`} variant="outlined" />}
        {node.artifactCount > 0 && <Chip size="small" label={`${node.artifactCount} artifacts`} variant="outlined" />}
        {node.warningCount > 0 && <Chip size="small" color="warning" label={`${node.warningCount} warnings`} />}
        {node.errorCount > 0 && <Chip size="small" color="error" label={`${node.errorCount} errors`} />}
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
          <DetailLine label="Route reason" value={node.routeReason} />
          {node.status !== 'skipped' && <DetailLine label="Skip reason" value={skipReason} />}
          <DetailLine label="Execution plan" value={node.executionPlan?.length ? node.executionPlan.join(' -> ') : undefined} />
          {(node.route || node.routeReason || node.executionPlan?.length || hasValue(node.llmResultSummary)) && (
            <InspectorSection title="Decision">
              <DetailLine label="Route" value={node.route} />
              <DetailLine label="Reason" value={node.routeReason} />
              <DetailLine label="Execution plan" value={node.executionPlan?.length ? node.executionPlan.join(' -> ') : undefined} />
              <TraceObject value={node.llmResultSummary} />
            </InspectorSection>
          )}
          {(hasValue(node.inputPreview) || hasValue(node.inputRefs)) && (
            <InspectorSection title="Input">
              <TraceObject value={node.inputPreview} />
              <TraceObject value={node.inputRefs} />
            </InspectorSection>
          )}
          {hasValue(node.promptSummary) && (
            <InspectorSection title="Prompt">
              <DetailLine label="Section" value={typeof node.promptSummary?.section === 'string' ? node.promptSummary.section : undefined} />
              <DetailLine label="Prompt chars" value={typeof node.promptSummary?.prompt_chars === 'number' ? node.promptSummary.prompt_chars : undefined} />
              <TraceObject value={node.promptSummary} />
            </InspectorSection>
          )}
          {node.toolSummaries.length > 0 && (
            <InspectorSection title="Tools">
              {node.toolSummaries.map((tool, index) => {
                const toolElapsed = formatDurationMs(tool.elapsedMs);
                return (
                  <Box key={`${tool.toolName}-${index}`} sx={{ mt: 0.75 }}>
                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                      {tool.displayName || tool.toolName}: {tool.ok ? 'ok' : 'failed'}
                      {toolElapsed ? `, ${toolElapsed}` : ''}
                      {tool.sourceCount ? `, sources ${tool.sourceCount}` : ''}
                      {tool.artifactKeys.length ? `, artifacts ${tool.artifactKeys.length}` : ''}
                      {tool.warnings.length ? `, warnings ${tool.warnings.join(', ')}` : ''}
                    </Typography>
                    {hasValue(tool.toolInput) && <DetailLine label="Input" value={<JsonPreview value={tool.toolInput} />} />}
                    {tool.resultPreview && <DetailLine label="Result preview" value={tool.resultPreview} />}
                    {hasValue(tool.artifactSummary) && <DetailLine label="Artifacts" value={<JsonPreview value={tool.artifactSummary} />} />}
                    {hasValue(tool.artifactRefs) && <TraceObject value={tool.artifactRefs} />}
                  </Box>
                );
              })}
            </InspectorSection>
          )}
          {(hasValue(node.outputPreview) || hasValue(node.outputRefs)) && (
            <InspectorSection title="Output">
              <TraceObject value={node.outputPreview} />
              <TraceObject value={node.outputRefs} />
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
        <InspectorSection title="Raw JSON">
          <JsonPreview value={rawPayload} maxHeight={360} />
        </InspectorSection>
      )}
    </Box>
  );
}
