import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import { formatDurationMs } from '../../lib/formatDuration';
import type { AgentRunDetails } from '../../lib/api';
import {
  getAvailableNodeCount,
  getAvailableToolCount,
  getRunDebugMetrics,
  getRunNodeEvents,
  getRunToolEvents,
  isSkippedNodeEvent,
} from './agent-debug-utils';
import { NodeEventsTooltip, ToolEventsTooltip } from './AgentRunTraceTooltips';

export default function AgentRunHeaderChips({ runDetails }: { runDetails: AgentRunDetails }) {
  const debug = runDetails.debug;
  const metrics = getRunDebugMetrics(runDetails);
  const formattedDuration = formatDurationMs(Number(metrics.duration_ms));
  const nodeEvents = getRunNodeEvents(runDetails);
  const toolEvents = getRunToolEvents(runDetails);
  const usedNodeCount = nodeEvents.filter((event) => !isSkippedNodeEvent(event)).length;
  const availableNodeCount = getAvailableNodeCount(runDetails);
  const usedToolCount = Number(metrics.tool_event_count ?? debug?.tool_event_count ?? toolEvents.length ?? 0);
  const availableToolCount = getAvailableToolCount(runDetails);
  const nodeCountLabel = availableNodeCount ? `${usedNodeCount}/${availableNodeCount}` : `${usedNodeCount}`;
  const toolCountLabel = availableToolCount ? `${usedToolCount}/${availableToolCount}` : `${usedToolCount}`;
  const warningCount = Number(metrics.tool_warning_count ?? debug?.tool_warning_count ?? 0);
  const errorCount = Number(metrics.error_count ?? debug?.error_count ?? metrics.tool_error_count ?? debug?.tool_error_count ?? 0);

  return (
    <>
      <Chip size="small" label={`Status: ${runDetails.status}`} variant="outlined" />
      {metrics.route && <Chip size="small" label={`Route: ${metrics.route}`} variant="outlined" />}
      {formattedDuration && <Chip size="small" label={`Run: ${formattedDuration}`} variant="outlined" />}
      <Tooltip title={<NodeEventsTooltip events={nodeEvents} usedCount={usedNodeCount} availableCount={availableNodeCount} />} placement="top" arrow>
        <Chip size="small" label={`Nodes: ${nodeCountLabel}`} variant="outlined" />
      </Tooltip>
      <Tooltip title={<ToolEventsTooltip events={toolEvents} />} placement="top" arrow>
        <Chip size="small" label={`Tools: ${toolCountLabel}`} variant="outlined" />
      </Tooltip>
      <Chip
        size="small"
        color={Number.isFinite(warningCount) && warningCount > 0 ? 'warning' : 'default'}
        label={`Warnings: ${Number.isFinite(warningCount) ? warningCount : 0}`}
        variant="outlined"
      />
      <Chip
        size="small"
        color={Number.isFinite(errorCount) && errorCount > 0 ? 'error' : 'default'}
        label={`Errors: ${Number.isFinite(errorCount) ? errorCount : 0}`}
        variant="outlined"
      />
    </>
  );
}
