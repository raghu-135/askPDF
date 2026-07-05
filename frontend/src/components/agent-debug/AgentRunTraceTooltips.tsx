import React from 'react';
import { Box, Typography } from '@mui/material';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import {
  formatTraceError,
  getNodeEventName,
  getToolEventName,
  isSkippedNodeEvent,
} from './agent-debug-utils';

const TraceTooltipList = ({
  title,
  emptyText,
  children,
}: {
  title: string;
  emptyText: string;
  children: React.ReactNode;
}) => (
  <Box sx={{ maxWidth: 560, maxHeight: 340, overflow: 'auto', p: 0.25 }}>
    <Typography variant="caption" sx={{ display: 'block', fontWeight: 700, mb: 0.5 }}>
      {title}
    </Typography>
    {children || (
      <Typography variant="caption" sx={{ display: 'block', opacity: 0.8 }}>
        {emptyText}
      </Typography>
    )}
  </Box>
);

export const NodeEventsTooltip = ({
  events,
  usedCount,
  availableCount,
}: {
  events: Record<string, any>[];
  usedCount: number;
  availableCount?: number;
}) => {
  const skippedCount = events.filter(isSkippedNodeEvent).length;
  const title = [
    `Node events: ${events.length}`,
    `used: ${usedCount}${availableCount ? `/${availableCount}` : ''}`,
    skippedCount ? `skipped: ${skippedCount}` : null,
  ].filter(Boolean).join(' · ');

  return (
    <TraceTooltipList title={title} emptyText="No node events recorded.">
      {events.map((event, index) => {
        const elapsed = formatDurationMs(Number(event?.elapsed_ms));
        const status = event?.status || (event?.skipped ? 'skipped' : 'completed');
        const skipReason = formatSkipReason(event?.skip_reason);
        const error = formatTraceError(event?.error);
        return (
          <Box key={`${getNodeEventName(event)}-${index}`} sx={{ py: 0.5, borderTop: index ? '1px solid rgba(255,255,255,0.18)' : 0 }}>
            <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
              {index + 1}. {getNodeEventName(event)}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
              {[status, elapsed, event?.route ? `route ${event.route}` : null, skipReason].filter(Boolean).join(' · ')}
            </Typography>
            {event?.route_reason && (
              <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                {String(event.route_reason)}
              </Typography>
            )}
            {error && (
              <Typography variant="caption" sx={{ display: 'block', color: 'error.light' }}>
                {error}
              </Typography>
            )}
          </Box>
        );
      })}
    </TraceTooltipList>
  );
};

export const ToolEventsTooltip = ({ events }: { events: Record<string, any>[] }) => (
  <TraceTooltipList title="Tool events" emptyText="No tool calls recorded.">
    {events.map((event, index) => {
      const elapsed = formatDurationMs(Number(event?.elapsed_ms));
      const artifactCount = Array.isArray(event?.artifact_keys)
        ? event.artifact_keys.length
        : event?.artifact_summary && typeof event.artifact_summary === 'object'
          ? Object.keys(event.artifact_summary).length
          : 0;
      const warningCount = Array.isArray(event?.warnings) ? event.warnings.length : 0;
      const error = formatTraceError(event?.error);
      return (
        <Box key={`${getToolEventName(event)}-${index}`} sx={{ py: 0.5, borderTop: index ? '1px solid rgba(255,255,255,0.18)' : 0 }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            {index + 1}. {getToolEventName(event)}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
            {[
              event?.caller_node ? `from ${event.caller_node}` : null,
              event?.ok === false ? 'failed' : 'ok',
              elapsed,
              Number.isFinite(Number(event?.source_count)) ? `${Number(event.source_count)} sources` : null,
              artifactCount ? `${artifactCount} artifacts` : null,
              warningCount ? `${warningCount} warnings` : null,
            ].filter(Boolean).join(' · ')}
          </Typography>
          {event?.result_preview && (
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
              {String(event.result_preview)}
            </Typography>
          )}
          {error && (
            <Typography variant="caption" sx={{ display: 'block', color: 'error.light' }}>
              {error}
            </Typography>
          )}
        </Box>
      );
    })}
  </TraceTooltipList>
);
