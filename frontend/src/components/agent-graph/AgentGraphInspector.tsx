import React from 'react';
import { Box, Chip, Typography } from '@mui/material';
import type { AgentGraphSelection } from './agent-graph-types';
import { DetailLine, JsonPreview } from './AgentGraphInspectorPrimitives';

/**
 * Graph inspection is intentionally edge-only. Node invocation details live in
 * the matching Execution Progress row so users never have two detail surfaces.
 */
export default function AgentGraphInspector({ selection }: { selection: AgentGraphSelection }) {
  if (!selection || selection.kind !== 'edge') return null;

  const { edge } = selection;
  return (
    <Box sx={{ p: 1, borderRadius: 1, bgcolor: 'rgba(0,0,0,0.03)' }}>
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
