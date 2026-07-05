import React, { useState } from 'react';
import { Box, Chip, Divider, Tab, Tabs, Tooltip, Typography } from '@mui/material';
import { JsonView } from 'react-json-view-lite';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentGraphSelection } from './agent-graph-types';

const sectionBg = 'rgba(0,0,0,0.03)';
const jsonTreeStyles = {
  container: 'askpdf-json-view',
  childFieldsContainer: 'askpdf-json-view__children',
  basicChildStyle: 'askpdf-json-view__child',
  collapseIcon: 'askpdf-json-view__collapse',
  expandIcon: 'askpdf-json-view__expand',
  collapsedContent: 'askpdf-json-view__collapsed',
  label: 'askpdf-json-view__label',
  clickableLabel: 'askpdf-json-view__clickable-label',
  nullValue: 'askpdf-json-view__null',
  undefinedValue: 'askpdf-json-view__undefined',
  numberValue: 'askpdf-json-view__number',
  stringValue: 'askpdf-json-view__string',
  booleanValue: 'askpdf-json-view__boolean',
  otherValue: 'askpdf-json-view__other',
  punctuation: 'askpdf-json-view__punctuation',
  quotesForFieldNames: false,
  stringifyStringValues: true,
  ariaLables: {
    collapseJson: 'Collapse JSON node',
    expandJson: 'Expand JSON node',
  },
};
const shouldExpandJsonNode = (level: number) => level < 4;

const SECTION_HELP: Record<string, string> = {
  Decision: 'Route, execution plan, and LLM decision data produced by router or planner nodes.',
  Input: 'State, refs, and bounded previews available before this node ran.',
  Prompt: 'Rendered LLM prompt summary for nodes that call the model.',
  Tools: 'Tool calls made by this worker node, including inputs, results, refs, and warnings.',
  Output: 'State, refs, and bounded previews produced by this node.',
  Warnings: 'Real warning codes from tool/runtime contracts. Skips and planner notes are not warnings.',
  'Raw JSON': 'Persisted node and tool event payloads used to render this inspector.',
};

const isJsonViewData = (value: unknown): value is Record<string, unknown> | unknown[] => {
  return value !== null && typeof value === 'object';
};

const JsonPreview = ({ value, maxHeight = 140 }: { value: unknown; maxHeight?: number }) => {
  const jsonSx = {
    '& .askpdf-json-view': {
      lineHeight: 1.35,
      whiteSpace: 'pre-wrap',
      overflowWrap: 'break-word',
      color: 'text.primary',
    },
    '& .askpdf-json-view__children': {
      m: 0,
      pl: 1.5,
      listStyle: 'none',
      borderLeft: '1px solid',
      borderColor: 'divider',
    },
    '& .askpdf-json-view__child': {
      m: 0,
      py: 0.1,
    },
    '& .askpdf-json-view__collapse, & .askpdf-json-view__expand': {
      display: 'inline-block',
      width: 14,
      mr: 0.5,
      cursor: 'pointer',
      userSelect: 'none',
      color: 'text.secondary',
    },
    '& .askpdf-json-view__collapse::after': {
      content: '"▾"',
    },
    '& .askpdf-json-view__expand::after': {
      content: '"▸"',
    },
    '& .askpdf-json-view__collapsed': {
      color: 'text.disabled',
      fontStyle: 'italic',
      mx: 0.5,
    },
    '& .askpdf-json-view__collapsed::after': {
      content: '"..."',
    },
    '& .askpdf-json-view__label, & .askpdf-json-view__clickable-label': {
      color: 'primary.main',
      fontWeight: 700,
      mr: 0.5,
    },
    '& .askpdf-json-view__clickable-label': {
      cursor: 'pointer',
    },
    '& .askpdf-json-view__string': {
      color: 'success.dark',
    },
    '& .askpdf-json-view__number': {
      color: 'secondary.main',
    },
    '& .askpdf-json-view__boolean': {
      color: 'warning.dark',
    },
    '& .askpdf-json-view__null, & .askpdf-json-view__undefined': {
      color: 'text.disabled',
      fontStyle: 'italic',
    },
    '& .askpdf-json-view__punctuation, & .askpdf-json-view__other': {
      color: 'text.secondary',
    },
  };

  return (
    <Box
      sx={{
        m: 0,
        mt: 0.5,
        p: 0.75,
        maxHeight,
        overflow: 'auto',
        borderRadius: 1,
        bgcolor: 'rgba(0,0,0,0.04)',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
        fontSize: '0.68rem',
        ...jsonSx,
      }}
    >
      {isJsonViewData(value) ? (
        <JsonView
          data={value}
          style={jsonTreeStyles}
          shouldExpandNode={shouldExpandJsonNode}
          clickToExpandNode
          compactTopLevel
        />
      ) : (
        <Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {JSON.stringify(value, null, 2)}
        </Box>
      )}
    </Box>
  );
};

const DetailLine = ({ label, value }: { label: string; value?: React.ReactNode }) => {
  if (!value) return null;
  return (
    <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
      <Box component="span" sx={{ fontWeight: 700, color: 'text.primary' }}>{label}: </Box>
      {value}
    </Typography>
  );
};

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => {
  if (!children) return null;
  return (
    <>
      <Divider sx={{ my: 1 }} />
      <Tooltip title={SECTION_HELP[title] || ''} placement="top" arrow>
        <Typography variant="caption" sx={{ display: 'inline-block', fontWeight: 700, cursor: 'help' }}>
          {title}
        </Typography>
      </Tooltip>
      {children}
    </>
  );
};

const hasValue = (value: unknown) => {
  if (!value) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length > 0;
  return true;
};

const TraceObject = ({ value }: { value: unknown }) => (
  hasValue(value) ? <JsonPreview value={value} /> : null
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
  const rawPayload = {
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
            <Section title="Decision">
              <DetailLine label="Route" value={node.route} />
              <DetailLine label="Reason" value={node.routeReason} />
              <DetailLine label="Execution plan" value={node.executionPlan?.length ? node.executionPlan.join(' -> ') : undefined} />
              <TraceObject value={node.llmResultSummary} />
            </Section>
          )}
          {(hasValue(node.inputPreview) || hasValue(node.inputRefs)) && (
            <Section title="Input">
              <TraceObject value={node.inputPreview} />
              <TraceObject value={node.inputRefs} />
            </Section>
          )}
          {hasValue(node.promptSummary) && (
            <Section title="Prompt">
              <DetailLine label="Section" value={typeof node.promptSummary?.section === 'string' ? node.promptSummary.section : undefined} />
              <DetailLine label="Prompt chars" value={typeof node.promptSummary?.prompt_chars === 'number' ? node.promptSummary.prompt_chars : undefined} />
              <TraceObject value={node.promptSummary} />
            </Section>
          )}
          {node.toolSummaries.length > 0 && (
            <Section title="Tools">
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
            </Section>
          )}
          {(hasValue(node.outputPreview) || hasValue(node.outputRefs)) && (
            <Section title="Output">
              <TraceObject value={node.outputPreview} />
              <TraceObject value={node.outputRefs} />
            </Section>
          )}
          {(node.warnings?.length || node.toolSummaries.some((tool) => tool.warnings.length > 0)) && (
            <Section title="Warnings">
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {(node.warnings || []).map((warning, index) => (
                  <Chip key={`node-warning-${index}`} size="small" color="warning" label={warning} />
                ))}
                {node.toolSummaries.flatMap((tool) => tool.warnings).map((warning, index) => (
                  <Chip key={`tool-warning-${index}`} size="small" color="warning" label={warning} variant="outlined" />
                ))}
              </Box>
            </Section>
          )}
        </>
      ) : (
        <Section title="Raw JSON">
          <JsonPreview value={rawPayload} maxHeight={360} />
        </Section>
      )}
    </Box>
  );
}
