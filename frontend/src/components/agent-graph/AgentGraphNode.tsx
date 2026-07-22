import React from 'react';
import AddIcon from '@mui/icons-material/Add';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { Box, Button, Chip, Tooltip, Typography } from '@mui/material';
import { Handle, NodeToolbar, Position } from '@xyflow/react';
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
  const visitCount = Number(data.visitCount || 0);
  const selectedVisitPosition = Number(data.selectedVisitPosition);
  const selectedVisitLabel = !data.authoring
    && Number.isInteger(selectedVisitPosition)
    && selectedVisitPosition > 0
    && visitCount > 0
    ? `Visit ${selectedVisitPosition} of ${visitCount}`
    : null;
  const tooltip = [
    data.label,
    data.description,
    instanceLabel !== data.label ? instanceLabel : null,
    isFocused ? 'focused by message' : null,
    visitCount > 1 ? `visits: ${visitCount}` : null,
    selectedVisitLabel ? `selected: ${selectedVisitLabel.toLowerCase()}` : null,
    data.route ? `route: ${data.route}` : null,
    elapsed ? `elapsed: ${elapsed}` : null,
    data.skipped ? skipReason || 'Skipped' : null,
    toolCount ? `tools: ${toolCount}` : null,
    data.warningCount ? `warnings: ${data.warningCount}` : null,
    data.errorCount ? `errors: ${data.errorCount}` : null,
    data.compatible === false ? data.compatibilityReason || 'This connection is not compatible.' : null,
  ].filter(Boolean).join('\n');
  const categoryColors: Record<string, string> = {
    context: '#5c6bc0',
    control: '#7e57c2',
    retrieval: '#1976d2',
    memory: '#0288d1',
    timeline: '#00897b',
    web: '#1565c0',
    answer: '#2e7d32',
    human_review: '#ed6c02',
    note: '#f9a825',
  };
  const categoryColor = categoryColors[data.category || ''] || '#607d8b';
  const outputPorts = data.authoring ? (data.outputPorts || [{ id: 'default', label: data.outputLabel || 'Next' }]) : [];
  const canAddBefore = data.authoring && data.id !== 'START' && Boolean(data.onAddPrevious);
  const canAddAfter = data.authoring && data.id !== 'END' && Boolean(data.onAddNext);
  const handleColor = data.compatible === false ? '#9e9e9e' : categoryColor;

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
          borderColor: data.compatible === false ? 'action.disabled' : selected || isFocused ? 'primary.main' : data.authoring ? categoryColor : statusColor[data.status] || 'divider',
          bgcolor: statusBg[data.status] || 'background.paper',
          boxShadow: selected || isFocused ? 3 : 1,
          color: 'text.primary',
          cursor: 'pointer',
          opacity: data.compatible === false ? 0.38 : 1,
          position: 'relative',
        }}
      >
        {data.authoring ? (
          <Handle
            id="input"
            type="target"
            position={targetPosition}
            aria-label={`Connect into ${data.label}`}
            title={`Connect into ${data.label}`}
            style={{
              width: 30,
              height: 48,
              left: isVertical ? undefined : -15,
              top: isVertical ? -24 : '50%',
              border: 0,
              borderRadius: 12,
              background: `radial-gradient(circle, ${handleColor} 0 7px, white 8px 9px, transparent 10px)`,
            }}
          />
        ) : <Handle type="target" position={targetPosition} style={{ opacity: 0 }} />}
        {canAddBefore || canAddAfter ? (
          <NodeToolbar position={Position.Top}>
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              {canAddBefore ? (
                <Tooltip title="Add a compatible previous step">
                  <Button
                    size="small"
                    aria-label="Add a compatible previous step"
                    startIcon={<ArrowBackIcon fontSize="small" />}
                    onClick={() => data.onAddPrevious?.(data.id)}
                    sx={{ minWidth: 0, px: 0.75, bgcolor: 'background.paper', boxShadow: 2 }}
                  >
                    <AddIcon fontSize="small" /> Before
                  </Button>
                </Tooltip>
              ) : null}
              {canAddAfter ? (
                <Tooltip title="Add a compatible next step">
                  <Button
                    size="small"
                    aria-label="Add a compatible next step"
                    endIcon={<ArrowForwardIcon fontSize="small" />}
                    onClick={() => data.onAddNext?.(data.id)}
                    sx={{ minWidth: 0, px: 0.75, bgcolor: 'background.paper', boxShadow: 2 }}
                  >
                    <AddIcon fontSize="small" /> After
                  </Button>
                </Tooltip>
              ) : null}
            </Box>
          </NodeToolbar>
        ) : null}
        {data.authoring && data.category ? (
          <Typography variant="overline" sx={{ color: categoryColor, fontWeight: 800, lineHeight: 1 }}>
            {String(data.category).replace(/_/g, ' ')}
          </Typography>
        ) : null}
        <Typography variant="body2" sx={{ display: 'block', fontWeight: 700, lineHeight: 1.25 }}>
          {data.label}
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', lineHeight: 1.3, mt: 0.1 }}>
          {instanceLabel}
        </Typography>
        {(data.authoring || data.type === 'canvas_note') && data.description ? (
          <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: 0.5, lineHeight: 1.25 }}>
            {data.description}
          </Typography>
        ) : null}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.4, mt: 0.8 }}>
          {data.authoring && data.usesLlm ? <Chip size="small" label="LLM" variant="outlined" sx={{ height: 22 }} /> : null}
          {data.authoring && data.usesTools ? <Chip size="small" label="Tool" variant="outlined" sx={{ height: 22 }} /> : null}
          {data.authoring && data.issueCount ? <Chip size="small" color="error" label={`${data.issueCount} issue${data.issueCount === 1 ? '' : 's'}`} sx={{ height: 22 }} /> : null}
          <Chip size="small" label={statusLabel} sx={{ height: 22, fontSize: '0.72rem' }} />
          {isFocused ? <Chip size="small" color="primary" label="focused" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {selectedVisitLabel ? (
            <Chip size="small" color="primary" label={selectedVisitLabel} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} />
          ) : visitCount > 1 ? (
            <Chip size="small" label={`visits ${visitCount}`} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} />
          ) : null}
          {elapsed && <Chip size="small" label={elapsed} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} />}
          {data.executionPlan?.length ? <Chip size="small" label={`plan ${data.executionPlan.length}`} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {toolCount ? <Chip size="small" label={`tools ${toolCount}`} variant="outlined" sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {data.warningCount ? <Chip size="small" color="warning" label={`warn ${data.warningCount}`} sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
          {data.errorCount ? <Chip size="small" color="error" label={`err ${data.errorCount}`} sx={{ height: 22, fontSize: '0.72rem' }} /> : null}
        </Box>
        {data.authoring ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.45, mt: 1, pr: 0.5 }}>
            {outputPorts.map((port, index) => (
              <Box key={port.id} sx={{ position: 'relative', display: 'flex', justifyContent: 'flex-end', alignItems: 'center', minHeight: 18 }}>
                <Tooltip title={port.description || ''}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', pr: 0.5 }}>{port.label}</Typography>
                </Tooltip>
                <Handle
                  id={port.id}
                  type="source"
                  position={sourcePosition}
                  style={{
                    width: 30,
                    height: 28,
                    top: isVertical ? undefined : '50%',
                    right: isVertical ? undefined : -15,
                    bottom: isVertical ? -14 : undefined,
                    left: isVertical ? `${20 + (index * 60) / Math.max(1, outputPorts.length)}%` : undefined,
                    transform: isVertical ? undefined : 'translateY(-50%)',
                    border: 0,
                    borderRadius: 12,
                    background: `radial-gradient(circle, ${categoryColor} 0 7px, white 8px 9px, transparent 10px)`,
                  }}
                  aria-label={`Connect from ${data.label}: ${port.label}`}
                  title={`Connect from ${data.label}: ${port.label}`}
                />
              </Box>
            ))}
          </Box>
        ) : <Handle type="source" position={sourcePosition} style={{ opacity: 0 }} />}
      </Box>
    </Tooltip>
  );
}
