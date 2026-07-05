import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentRunDetails } from '../../lib/api';
import type { TraceRunView } from './agent-trace-projection';
import { TraceNodesTooltip, TraceToolsTooltip } from './AgentRunTraceTooltips';

export default function AgentRunHeaderChips({
  runDetails,
  traceView,
}: {
  runDetails: AgentRunDetails;
  traceView: TraceRunView;
}) {
  const { metrics } = traceView;
  const formattedDuration = formatDurationMs(Number(metrics.duration_ms));
  const nodeCountLabel = traceView.availableNodeCount ? `${traceView.usedNodeCount}/${traceView.availableNodeCount}` : `${traceView.usedNodeCount}`;
  const toolCountLabel = traceView.availableToolCount ? `${traceView.usedToolCount}/${traceView.availableToolCount}` : `${traceView.usedToolCount}`;

  return (
    <>
      <Chip size="small" label={`Status: ${runDetails.status}`} variant="outlined" />
      {traceView.route && <Chip size="small" label={`Route: ${traceView.route}`} variant="outlined" />}
      {formattedDuration && <Chip size="small" label={`Run: ${formattedDuration}`} variant="outlined" />}
      <Tooltip title={<TraceNodesTooltip nodes={traceView.nodes} usedCount={traceView.usedNodeCount} availableCount={traceView.availableNodeCount} />} placement="top" arrow>
        <Chip size="small" label={`Nodes: ${nodeCountLabel}`} variant="outlined" />
      </Tooltip>
      <Tooltip title={<TraceToolsTooltip tools={traceView.tools} />} placement="top" arrow>
        <Chip size="small" label={`Tools: ${toolCountLabel}`} variant="outlined" />
      </Tooltip>
      <Chip
        size="small"
        color={traceView.warningCount > 0 ? 'warning' : 'default'}
        label={`Warnings: ${traceView.warningCount}`}
        variant="outlined"
      />
      <Chip
        size="small"
        color={traceView.errorCount > 0 ? 'error' : 'default'}
        label={`Errors: ${traceView.errorCount}`}
        variant="outlined"
      />
    </>
  );
}
