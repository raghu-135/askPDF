import React from 'react';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import { Alert, Box, Button, Chip, Paper, Stack, Typography } from '@mui/material';
import type { AgentTraceDiagnostics, AgentTraceFailure, AgentTraceLocation } from '../../lib/api';

const readable = (value: string) => value.replaceAll('_', ' ');

const locationLabel = (location: AgentTraceLocation) => (
  location.operation_label
  || location.operation_id
  || location.tool_name
  || location.subagent_id
  || location.approval_id
  || 'Run lifecycle'
);

export default function TraceDiagnosticsPanel({
  diagnostics,
  onShowEvent,
  onOpenOperation,
}: {
  diagnostics: AgentTraceDiagnostics;
  onShowEvent: (eventId: string) => void;
  onOpenOperation: (operationId: string, attempt?: number) => void;
}) {
  const { summary, failures, groups, observability_gaps: gaps } = diagnostics;
  if (summary.failure_count === 0 && summary.cancellation_count === 0 && gaps.length === 0) return null;
  const primary = failures.find((failure) => failure.event_id === summary.primary_failure_event_id);
  return (
    <Stack spacing={0.75} sx={{ mb: 0.75 }}>
      <Alert severity="error" icon={<ErrorOutlineIcon />} sx={{ '& .MuiAlert-message': { width: '100%' } }}>
        <Typography variant="overline" sx={{ lineHeight: 1.2 }}>What failed</Typography>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{readable(summary.code)}</Typography>
        <Typography variant="body2">{summary.message}</Typography>
        <Stack direction="row" spacing={0.5} flexWrap="wrap" sx={{ mt: 0.6, rowGap: 0.5 }}>
          <Chip size="small" label={locationLabel(summary.location)} />
          {summary.location.attempt && <Chip size="small" variant="outlined" label={`Attempt ${summary.location.attempt}`} />}
          <Chip size="small" variant="outlined" label={summary.retryable ? 'Retryable' : 'Not retryable'} />
          {summary.primary_basis && <Chip size="small" variant="outlined" label={summary.primary_basis === 'explicit_cause' ? 'Explicit cause' : 'Earliest observed'} />}
          {primary?.occurred_at && <Chip size="small" variant="outlined" label={new Date(primary.occurred_at).toLocaleTimeString()} />}
        </Stack>
        {primary && <Stack direction="row" spacing={0.5} sx={{ mt: 0.6 }}>
          <Button size="small" onClick={() => onShowEvent(primary.event_id)}>Show event</Button>
          {primary.location.operation_id && <Button size="small" onClick={() => onOpenOperation(primary.location.operation_id!, primary.location.attempt)}>Open operation</Button>}
        </Stack>}
      </Alert>

      {gaps.map((gap) => <Alert key={gap.code} severity="warning">{gap.message}</Alert>)}

      {failures.length > 1 && <Paper variant="outlined" sx={{ p: 0.75 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Correlated failures ({failures.length})</Typography>
        <Stack spacing={0.5} sx={{ mt: 0.5 }}>
          {failures.map((failure: AgentTraceFailure) => (
            <Box key={failure.event_id} component="details" open={failure.classification === 'primary'}>
              <Box component="summary" sx={{ cursor: 'pointer' }}>
                <Stack component="span" direction="row" spacing={0.5} alignItems="center" sx={{ display: 'inline-flex', flexWrap: 'wrap' }}>
                  <Chip size="small" color={failure.classification === 'primary' || failure.classification === 'terminal_summary' ? 'error' : 'default'} label={readable(failure.classification)} />
                  <Typography variant="caption" sx={{ fontWeight: 700 }}>{readable(failure.code)}</Typography>
                  <Typography variant="caption" color="text.secondary">· {locationLabel(failure.location)}</Typography>
                </Stack>
              </Box>
              <Typography variant="caption" component="div" sx={{ mt: 0.35 }}>{failure.message}</Typography>
              {failure.caused_by_event_id && <Typography variant="caption" component="div" color="text.secondary">Caused by event {failure.caused_by_event_id}</Typography>}
              <Stack direction="row" spacing={0.5} sx={{ mt: 0.35 }}>
                <Button size="small" onClick={() => onShowEvent(failure.event_id)}>
                  {failure.location.tool_name ? 'Show tool event' : failure.location.subagent_id ? 'Show subagent event' : 'Show event'}
                </Button>
                {failure.location.operation_id && <Button size="small" onClick={() => onOpenOperation(failure.location.operation_id!, failure.location.attempt)}>Open operation</Button>}
              </Stack>
            </Box>
          ))}
        </Stack>
      </Paper>}

      {groups.some((group) => group.occurrence_count > 1) && <Paper variant="outlined" sx={{ p: 0.75 }}>
        <Typography variant="caption" sx={{ fontWeight: 700 }}>Repeated failure groups</Typography>
        {groups.filter((group) => group.occurrence_count > 1).map((group) => (
          <Typography key={`${group.code}:${group.event_ids[0]}`} variant="caption" component="div" color="text.secondary">
            {readable(group.code)} · {locationLabel(group.location)} · {group.occurrence_count} occurrences
          </Typography>
        ))}
      </Paper>}
    </Stack>
  );
}
