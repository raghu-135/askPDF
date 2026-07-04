import React from 'react';
import { Box, Chip, Divider, Typography } from '@mui/material';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentGraphSelection } from './agent-graph-types';

const sectionBg = 'rgba(0,0,0,0.03)';

const JsonPreview = ({ value }: { value: unknown }) => (
  <Box
    component="pre"
    sx={{
      m: 0,
      mt: 0.5,
      p: 0.75,
      maxHeight: 140,
      overflow: 'auto',
      borderRadius: 1,
      bgcolor: 'rgba(0,0,0,0.04)',
      fontSize: '0.68rem',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
    }}
  >
    {JSON.stringify(value, null, 2)}
  </Box>
);

const DetailLine = ({ label, value }: { label: string; value?: React.ReactNode }) => {
  if (!value) return null;
  return (
    <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
      <Box component="span" sx={{ fontWeight: 700, color: 'text.primary' }}>{label}: </Box>
      {value}
    </Typography>
  );
};

export default function AgentGraphInspector({ selection }: { selection: AgentGraphSelection }) {
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
      <DetailLine label="Route reason" value={node.routeReason} />
      {node.status !== 'skipped' && <DetailLine label="Skip reason" value={skipReason} />}
      <DetailLine label="Execution plan" value={node.executionPlan?.length ? node.executionPlan.join(' -> ') : undefined} />
      {node.toolSummaries.length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            Tools
          </Typography>
          {node.toolSummaries.map((tool, index) => {
            const toolElapsed = formatDurationMs(tool.elapsedMs);
            return (
              <Typography key={`${tool.toolName}-${index}`} variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                {tool.displayName || tool.toolName}: {tool.ok ? 'ok' : 'failed'}
                {toolElapsed ? `, ${toolElapsed}` : ''}
                {tool.sourceCount ? `, sources ${tool.sourceCount}` : ''}
                {tool.artifactKeys.length ? `, artifacts ${tool.artifactKeys.length}` : ''}
                {tool.warnings.length ? `, warnings ${tool.warnings.join(', ')}` : ''}
              </Typography>
            );
          })}
        </>
      )}
      {node.rawEvents.length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            Node Events
          </Typography>
          <JsonPreview value={node.rawEvents} />
        </>
      )}
    </Box>
  );
}
