import React from 'react';
import { Box, Chip, Tooltip, Typography } from '@mui/material';
import { Handle, Position } from '@xyflow/react';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentGraphNode as AgentGraphNodeModel } from './agent-graph-types';

const statusColor: Record<string, string> = {
  active: '#2e7d32',
  planned: '#1565c0',
  skipped: '#757575',
  inactive: '#9e9e9e',
  error: '#c62828',
};

const statusBg: Record<string, string> = {
  active: 'rgba(46, 125, 50, 0.12)',
  planned: 'rgba(21, 101, 192, 0.12)',
  skipped: 'rgba(117, 117, 117, 0.12)',
  inactive: 'rgba(158, 158, 158, 0.08)',
  error: 'rgba(198, 40, 40, 0.12)',
};

export default function AgentGraphNode({ data, selected }: { data: AgentGraphNodeModel; selected?: boolean }) {
  const elapsed = formatDurationMs(data.elapsedMs);
  const toolCount = data.toolSummaries.length;
  const isVertical = data.layoutDirection === 'DOWN';
  const targetPosition = isVertical ? Position.Top : Position.Left;
  const sourcePosition = isVertical ? Position.Bottom : Position.Right;
  const skipReason = formatSkipReason(data.skipReason);
  const statusLabel = data.status === 'skipped' ? skipReason || 'Skipped' : data.status;
  const isFocused = data.focused === true;
  const instanceLabel = data.instanceLabel || data.id;
  const tooltip = [
    data.label,
    instanceLabel !== data.label ? instanceLabel : null,
    isFocused ? 'focused by message' : null,
    data.route ? `route: ${data.route}` : null,
    elapsed ? `elapsed: ${elapsed}` : null,
    data.skipped ? skipReason || 'Skipped' : null,
    toolCount ? `tools: ${toolCount}` : null,
    data.warningCount ? `warnings: ${data.warningCount}` : null,
    data.errorCount ? `errors: ${data.errorCount}` : null,
  ].filter(Boolean).join('\n');

  return (
    <Tooltip title={<Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap' }}>{tooltip}</Box>} arrow>
      <Box
        sx={{
          minWidth: 205,
          maxWidth: 230,
          px: 1.25,
          py: 1,
          borderRadius: 1,
          border: isFocused ? 2 : 1,
          borderColor: selected || isFocused ? 'primary.main' : statusColor[data.status] || 'divider',
          bgcolor: statusBg[data.status] || 'background.paper',
          boxShadow: selected || isFocused ? 3 : 1,
          color: 'text.primary',
          cursor: 'pointer',
        }}
      >
        <Handle type="target" position={targetPosition} style={{ opacity: 0 }} />
        <Typography variant="body2" sx={{ display: 'block', fontWeight: 700, lineHeight: 1.25 }}>
          {data.label}
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', lineHeight: 1.3, mt: 0.1 }}>
          {instanceLabel}
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.4, mt: 0.8 }}>
          <Chip size="small" label={statusLabel} sx={{ height: 22, fontSize: '0.72rem' }} />
          {isFocused ? <Chip size="small" color="primary" label="focused" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {elapsed && <Chip size="small" label={elapsed} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} />}
          {data.executionPlan?.length ? <Chip size="small" label={`plan ${data.executionPlan.length}`} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {toolCount ? <Chip size="small" label={`tools ${toolCount}`} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {data.warningCount ? <Chip size="small" color="warning" label={`warn ${data.warningCount}`} sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {data.errorCount ? <Chip size="small" color="error" label={`err ${data.errorCount}`} sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
        </Box>
        <Handle type="source" position={sourcePosition} style={{ opacity: 0 }} />
      </Box>
    </Tooltip>
  );
}
