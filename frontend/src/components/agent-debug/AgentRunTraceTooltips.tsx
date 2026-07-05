import React from 'react';
import { Box, Typography } from '@mui/material';
import { formatSkipReason } from '../../lib/agentDebugLabels';
import { formatDurationMs } from '../../lib/formatDuration';
import { formatTraceError } from './agent-debug-utils';
import type { TraceNodeView, TraceToolView } from './agent-trace-projection';

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

export const TraceNodesTooltip = ({
  nodes,
  usedCount,
  availableCount,
}: {
  nodes: TraceNodeView[];
  usedCount: number;
  availableCount?: number;
}) => {
  const skippedCount = nodes.filter((node) => node.skipped).length;
  const title = [
    `Node spans: ${nodes.length}`,
    `used: ${usedCount}${availableCount ? `/${availableCount}` : ''}`,
    skippedCount ? `skipped: ${skippedCount}` : null,
  ].filter(Boolean).join(' · ');

  return (
    <TraceTooltipList title={title} emptyText="No node spans recorded.">
      {nodes.map((node, index) => {
        const elapsed = formatDurationMs(node.durationMs);
        const status = node.status || (node.skipped ? 'skipped' : 'completed');
        const skipReason = formatSkipReason(node.raw?.skip_reason);
        const error = formatTraceError(node.error);
        return (
          <Box key={`${node.id}-${index}`} sx={{ py: 0.5, borderTop: index ? '1px solid rgba(255,255,255,0.18)' : 0 }}>
            <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
              {index + 1}. {node.id}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
              {[status, elapsed, node.route ? `route ${node.route}` : null, skipReason].filter(Boolean).join(' · ')}
            </Typography>
            {node.routeReason && (
              <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                {node.routeReason}
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

export const TraceToolsTooltip = ({ tools }: { tools: TraceToolView[] }) => (
  <TraceTooltipList title="Tool spans" emptyText="No tool calls recorded.">
    {tools.map((tool, index) => {
      const elapsed = formatDurationMs(tool.durationMs);
      const raw = tool.raw || {};
      const artifactCount = Array.isArray(raw.artifact_keys)
        ? raw.artifact_keys.length
        : raw.artifact_summary && typeof raw.artifact_summary === 'object'
          ? Object.keys(raw.artifact_summary).length
          : 0;
      const warningCount = tool.warningCodes.length;
      const error = formatTraceError(raw.error);
      return (
        <Box key={`${tool.name}-${index}`} sx={{ py: 0.5, borderTop: index ? '1px solid rgba(255,255,255,0.18)' : 0 }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            {index + 1}. {tool.displayName || tool.name}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
            {[
              tool.callerNode ? `from ${tool.callerNode}` : null,
              tool.ok ? 'ok' : 'failed',
              elapsed,
              Number.isFinite(Number(tool.sourceCount)) ? `${Number(tool.sourceCount)} sources` : null,
              artifactCount ? `${artifactCount} artifacts` : null,
              warningCount ? `${warningCount} warnings` : null,
            ].filter(Boolean).join(' · ')}
          </Typography>
          {raw.result_preview && (
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
              {String(raw.result_preview)}
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
