import React from 'react';
import { Box, Button, Chip, Stack, Tooltip, Typography } from '@mui/material';
import type { AgentTraceParallelAttempt, AgentTraceParallelGroup } from '../../lib/api';

const statusColor = (status: string) => {
  if (['failed', 'timed_out'].includes(status)) return 'error.main';
  if (status === 'cancelled') return 'warning.main';
  if (status === 'completed') return 'success.main';
  if (status === 'skipped') return 'text.disabled';
  return 'info.main';
};

const attemptRange = (group: AgentTraceParallelGroup, attempt: AgentTraceParallelAttempt) => {
  const groupStart = group.started_at ? Date.parse(group.started_at) : Number.NaN;
  const groupEnd = group.completed_at ? Date.parse(group.completed_at) : Number.NaN;
  const attemptStart = attempt.started_at ? Date.parse(attempt.started_at) : Number.NaN;
  const attemptEnd = attempt.completed_at ? Date.parse(attempt.completed_at) : Number.NaN;
  if ([groupStart, groupEnd, attemptStart, attemptEnd].every(Number.isFinite) && groupEnd > groupStart) {
    return {
      left: Math.max(0, Math.min(100, ((attemptStart - groupStart) / (groupEnd - groupStart)) * 100)),
      width: Math.max(2, Math.min(100, ((attemptEnd - attemptStart) / (groupEnd - groupStart)) * 100)),
    };
  }
  const span = Math.max(1, group.last_sequence - group.first_sequence + 1);
  return {
    left: Math.max(0, ((attempt.first_sequence - group.first_sequence) / span) * 100),
    width: Math.max(2, ((attempt.last_sequence - attempt.first_sequence + 1) / span) * 100),
  };
};

export default function ParallelTraceLanes({
  groups,
  onEventFocus,
  onOperationFocus,
  operationLabels = {},
}: {
  groups: AgentTraceParallelGroup[];
  onEventFocus?: (eventId: string) => void;
  onOperationFocus?: (operationId: string) => void;
  operationLabels?: Record<string, string>;
}) {
  return (
    <Stack spacing={1} sx={{ py: 0.5 }}>
      {groups.map((group) => (
        <Box key={group.group_id} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 0.75 }}>
          <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap">
            <Typography variant="caption" sx={{ fontWeight: 700 }}>Dispatch {group.group_id}</Typography>
            <Chip size="small" variant="outlined" label={group.status} sx={{ height: 20 }} />
            <Typography variant="caption" color="text.secondary">{group.members.length}/{group.planned} workers</Typography>
            <Typography variant="caption" color="text.secondary">barrier {group.barrier.event_id ? group.barrier.status : group.status === 'running' ? 'pending' : 'not reported'}</Typography>
            <Typography variant="caption" color="text.secondary">aggregation {group.aggregation.event_id ? group.aggregation.status : group.status === 'running' ? 'pending' : 'not reported'}</Typography>
          </Stack>
          <Stack spacing={0.5} sx={{ mt: 0.65 }}>
            {group.members.map((member) => (
              <Stack key={member.member_id} direction="row" spacing={0.75} alignItems="center">
                <Button
                  size="small"
                  variant="text"
                  title={member.member_id}
                  onClick={() => {
                    const eventId = member.event_ids[member.event_ids.length - 1];
                    if (eventId) onEventFocus?.(eventId);
                    if (member.operation_id) onOperationFocus?.(member.operation_id);
                  }}
                  sx={{ minWidth: 140, maxWidth: 220, justifyContent: 'flex-start', textTransform: 'none', fontSize: '0.72rem' }}
                >
                  {member.operation_label || member.tool_name || member.subagent_id || (member.operation_id ? operationLabels[member.operation_id] || member.operation_id : member.member_id)}
                </Button>
                <Box sx={{ position: 'relative', flex: 1, minWidth: 120, height: 18, bgcolor: 'action.hover', borderRadius: 0.5, overflow: 'hidden' }}>
                  {member.attempts.map((attempt) => {
                    const range = attemptRange(group, attempt);
                    const eventId = attempt.failure_event_ids[0] || attempt.event_ids[attempt.event_ids.length - 1];
                    return (
                      <Tooltip key={attempt.attempt} title={`Attempt ${attempt.attempt} · ${attempt.status}${attempt.duration_ms != null ? ` · ${Math.round(attempt.duration_ms)} ms` : ''}`}>
                        <Box
                          role="button"
                          tabIndex={0}
                          onClick={() => eventId && onEventFocus?.(eventId)}
                          onKeyDown={(event) => { if ((event.key === 'Enter' || event.key === ' ') && eventId) onEventFocus?.(eventId); }}
                          sx={{ position: 'absolute', left: `${range.left}%`, width: `${range.width}%`, top: 2, bottom: 2, bgcolor: statusColor(attempt.status), borderRadius: 0.4, cursor: eventId ? 'pointer' : 'default' }}
                        />
                      </Tooltip>
                    );
                  })}
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ width: 72, textAlign: 'right' }}>{member.status}</Typography>
              </Stack>
            ))}
          </Stack>
        </Box>
      ))}
    </Stack>
  );
}
