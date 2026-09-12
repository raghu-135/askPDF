import React, { useEffect, useRef } from 'react';
import { Alert, Box, Chip, Paper, Stack, Typography } from '@mui/material';
import type { AgentTraceTimelineEvent } from '../../lib/api';

const eventLabel = (event: AgentTraceTimelineEvent) => {
  const payload = event.payload || {};
  return String(
    payload.operation_label
    || payload.tool_name
    || payload.title
    || payload.subagent_name
    || event.operation_id
    || event.kind,
  );
};

export default function GenericTraceTimeline({ events, focusedEventId, defaultOpen = false }: { events: AgentTraceTimelineEvent[]; focusedEventId?: string | null; defaultOpen?: boolean }) {
  const containerRef = useRef<HTMLDetailsElement | null>(null);
  const eventRefs = useRef(new Map<string, HTMLElement>());
  useEffect(() => {
    if (!focusedEventId) return;
    if (containerRef.current) containerRef.current.open = true;
    const frame = window.requestAnimationFrame(() => eventRefs.current.get(focusedEventId)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
    return () => window.cancelAnimationFrame(frame);
  }, [focusedEventId]);
  if (events.length === 0) return null;
  return (
    <Paper elevation={0} square sx={{ px: 1, py: 0.5, borderTop: 1, borderColor: 'divider', bgcolor: 'background.default' }}>
      <Box component="details" ref={containerRef} open={defaultOpen || undefined}>
        <Box component="summary" sx={{ cursor: 'pointer', py: 0.35, fontSize: '0.78rem', fontWeight: 700 }}>
          Raw canonical event journal ({events.length})
        </Box>
        <Stack spacing={0.5} sx={{ py: 0.5 }}>
          {[...events].sort((a, b) => a.sequence - b.sequence).map((event) => {
            const failed = event.kind.endsWith('.failed') || ['failed', 'failure', 'error'].includes(String(event.status || event.payload?.status || '').toLowerCase()) || Boolean(event.payload?.error);
            const details = { payload: event.payload || {}, framework_details: event.framework_details || {} };
            return (
              <Box component="details" key={event.event_id} open={failed || focusedEventId === event.event_id} ref={(node) => { if (node) eventRefs.current.set(event.event_id, node as HTMLElement); else eventRefs.current.delete(event.event_id); }} sx={{ minWidth: 0, outline: focusedEventId === event.event_id ? '1px solid' : 'none', outlineColor: 'primary.main' }}>
                <Box component="summary" sx={{ cursor: 'pointer', listStylePosition: 'inside' }}>
                  <Stack component="span" direction="row" spacing={0.75} alignItems="center" sx={{ display: 'inline-flex', width: 'calc(100% - 18px)', minWidth: 0 }}>
                    <Chip size="small" color={failed ? 'error' : 'default'} variant="outlined" label={event.kind} sx={{ height: 20, flexShrink: 0 }} />
                    <Typography variant="caption" noWrap title={eventLabel(event)}>{eventLabel(event)}</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>#{event.sequence}</Typography>
                  </Stack>
                </Box>
                <Alert severity={failed ? 'error' : 'info'} icon={false} sx={{ mt: 0.35, py: 0.25, '& .MuiAlert-message': { width: '100%' } }}>
                  <Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: '0.7rem' }}>
                    {JSON.stringify(details, null, 2)}
                  </Box>
                </Alert>
              </Box>
            );
          })}
        </Stack>
      </Box>
    </Paper>
  );
}
