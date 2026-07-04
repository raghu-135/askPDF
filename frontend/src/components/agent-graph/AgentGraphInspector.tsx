import React from 'react';
import { Box, Chip, Divider, Typography } from '@mui/material';
import type { AgentGraphSelection } from './agent-graph-types';

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

export default function AgentGraphInspector({ selection }: { selection: AgentGraphSelection }) {
  if (!selection) {
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
        <Typography variant="caption" color="text.secondary">
          Select a node or edge for details.
        </Typography>
      </Box>
    );
  }

  if (selection.kind === 'edge') {
    const { edge } = selection;
    return (
      <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
        <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
          Edge: {edge.source} {'->'} {edge.target}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
          {edge.label && <Chip size="small" label={`route ${edge.label}`} variant="outlined" />}
          <Chip size="small" label={edge.conditional ? 'conditional' : 'sequential'} variant="outlined" />
          <Chip size="small" label={edge.active ? 'active' : 'inactive'} variant="outlined" />
          {edge.selected && <Chip size="small" color="primary" label="selected" />}
        </Box>
        {edge.raw && <JsonPreview value={edge.raw} />}
      </Box>
    );
  }

  const { node } = selection;
  return (
    <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
      <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
        Node: {node.label}
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
        <Chip size="small" label={node.status} variant="outlined" />
        {node.elapsedMs && <Chip size="small" label={`${Math.round(node.elapsedMs)}ms`} variant="outlined" />}
        {node.route && <Chip size="small" label={`route ${node.route}`} variant="outlined" />}
        {node.warningCount > 0 && <Chip size="small" color="warning" label={`${node.warningCount} warnings`} />}
        {node.errorCount > 0 && <Chip size="small" color="error" label={`${node.errorCount} errors`} />}
      </Box>
      {node.routeReason && (
        <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
          Route reason: {node.routeReason}
        </Typography>
      )}
      {node.skipReason && (
        <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
          Skip reason: {node.skipReason}
        </Typography>
      )}
      {node.executionPlan?.length ? (
        <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
          Execution plan: {node.executionPlan.join(' -> ')}
        </Typography>
      ) : null}
      {node.toolSummaries.length > 0 && (
        <>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            Tools
          </Typography>
          {node.toolSummaries.map((tool, index) => (
            <Typography key={`${tool.toolName}-${index}`} variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              {tool.displayName || tool.toolName}: {tool.ok ? 'ok' : 'failed'}
              {tool.elapsedMs ? `, ${Math.round(tool.elapsedMs)}ms` : ''}
              {tool.warnings.length ? `, warnings ${tool.warnings.join(', ')}` : ''}
            </Typography>
          ))}
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
