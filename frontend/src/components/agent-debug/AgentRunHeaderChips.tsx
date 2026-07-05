import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentRunDetails } from '../../lib/api';
import type { TraceRunView } from './agent-trace-projection';
import { TraceLlmUsageTooltip, TraceNodesTooltip, TraceToolsTooltip } from './AgentRunTraceTooltips';

const formatTokenCount = (value: unknown) => {
  const count = Number(value);
  if (!Number.isFinite(count) || count <= 0) return undefined;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(count >= 10_000_000 ? 0 : 1)}m`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(count >= 10_000 ? 0 : 1)}k`;
  return count.toLocaleString();
};

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
  const tokenCountLabel = formatTokenCount(metrics.llm_token_count_total);

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
      {tokenCountLabel && (
        <Tooltip title={<TraceLlmUsageTooltip metrics={metrics} />} placement="top" arrow>
          <Chip size="small" label={`Tokens: ${tokenCountLabel}`} variant="outlined" />
        </Tooltip>
      )}
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
