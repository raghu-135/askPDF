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

const tokenValue = (value: unknown) => (
  typeof value === 'number' || typeof value === 'string' ? String(value) : undefined
);

const booleanLabel = (value: unknown) => (
  typeof value === 'boolean' ? (value ? 'yes' : 'no') : undefined
);

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
  const instanceLabel = node.instanceLabel || node.id;
  const llmSummary = node.llmSummary || {};
  const observability = node.observability && typeof node.observability === 'object'
    ? node.observability as Record<string, unknown>
    : {};
  const capabilities = Array.isArray(node.capabilities)
    ? node.capabilities.filter((item): item is string => typeof item === 'string' && item.length > 0)
    : [];
  const tokenCounts = llmSummary.token_counts && typeof llmSummary.token_counts === 'object'
    ? llmSummary.token_counts as Record<string, unknown>
    : {};
  const retryAttempts = Array.isArray(llmSummary.retry_attempts)
    ? llmSummary.retry_attempts.filter((attempt): attempt is Record<string, any> => attempt && typeof attempt === 'object')
    : [];
  const toolTraceSpans = node.toolSummaries
    .map((tool) => tool.traceSpan)
    .filter((span): span is Record<string, any> => Boolean(span));
  const rawPayload = {
    focused_span_ids: node.focusedSpanIds || [],
    focused_trace_spans: node.focusedTraceSpans || [],
    trace_spans: node.traceSpans || [],
    tool_trace_spans: toolTraceSpans,
    node_rows: node.rawEvents.map(withoutInternalTraceFields),
    tool_rows: node.toolSummaries.map((tool) => withoutInternalTraceFields(tool.raw)),
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
        {node.route && <Chip size="small" label={`route ${node.route}`} variant="outlined" />}
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
          {hasValue(node.focusedTraceSpans) && (
            <InspectorSection title="Focused Spans">
              <TraceObject value={node.focusedTraceSpans} />
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
          {hasValue(node.llmSummary) && (
            <InspectorSection title="LLM">
              <DetailLine label="Model" value={typeof llmSummary.model_name === 'string' ? llmSummary.model_name : undefined} />
              <DetailLine label="Response chars" value={tokenValue(llmSummary.response_chars)} />
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
                {tokenValue(tokenCounts.prompt) && <Chip size="small" variant="outlined" label={`prompt ${tokenValue(tokenCounts.prompt)}`} />}
                {tokenValue(tokenCounts.completion) && <Chip size="small" variant="outlined" label={`completion ${tokenValue(tokenCounts.completion)}`} />}
                {tokenValue(tokenCounts.total) && <Chip size="small" color="primary" variant="outlined" label={`total ${tokenValue(tokenCounts.total)}`} />}
                {tokenValue(tokenCounts.reasoning) && <Chip size="small" variant="outlined" label={`reasoning ${tokenValue(tokenCounts.reasoning)}`} />}
                {tokenValue(tokenCounts.cached) && <Chip size="small" variant="outlined" label={`cached ${tokenValue(tokenCounts.cached)}`} />}
                {tokenValue(llmSummary.retry_count) && <Chip size="small" color={Number(llmSummary.retry_count) > 0 ? 'warning' : 'default'} variant="outlined" label={`retries ${tokenValue(llmSummary.retry_count)}`} />}
              </Box>
              <DetailLine label="Reasoning available" value={booleanLabel(llmSummary.reasoning_available)} />
              <DetailLine label="Reasoning format" value={typeof llmSummary.reasoning_format === 'string' ? llmSummary.reasoning_format : undefined} />
              <DetailLine label="Reasoning chars" value={tokenValue(llmSummary.reasoning_chars)} />
              {typeof llmSummary.reasoning_preview === 'string' && (
                <DetailLine
                  label="Reasoning preview"
                  value={(
                    <Box
                      component="span"
                      sx={{
                        display: 'block',
                        mt: 0.5,
                        p: 0.75,
                        borderRadius: 1,
                        bgcolor: 'rgba(0,0,0,0.04)',
                        color: 'text.primary',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {llmSummary.reasoning_preview}
                    </Box>
                  )}
                />
              )}
              {retryAttempts.length > 0 && (
                <Box sx={{ mt: 0.75 }}>
                  {retryAttempts.map((attempt, index) => (
                    <Typography key={`llm-retry-${index}`} variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                      Retry {attempt.attempt ?? index + 1}
                      {attempt.delay_ms ? `, delay ${attempt.delay_ms}ms` : ''}
                      {attempt.http_status_code ? `, HTTP ${attempt.http_status_code}` : ''}
                      {attempt.reason ? `, ${attempt.reason}` : ''}
                      {attempt.exception_type ? `, ${attempt.exception_type}` : ''}
                    </Typography>
                  ))}
                </Box>
              )}
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
