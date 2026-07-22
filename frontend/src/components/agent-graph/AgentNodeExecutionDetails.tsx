import React from 'react';
import { Alert, Box, Divider, Stack, Typography } from '@mui/material';
import type { AgentRunNodeDetail } from '../../lib/api';
import { JsonPreview } from './AgentGraphInspectorPrimitives';

const Section = ({ title, children, defaultOpen = true }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) => (
  <Box component="details" open={defaultOpen} sx={{ mt: 1 }}>
    <Box component="summary" sx={{ cursor: 'pointer', fontSize: '0.75rem', fontWeight: 700 }}>{title}</Box>
    <Box sx={{ mt: 0.5 }}>{children}</Box>
  </Box>
);

const hasData = (value: unknown) => value !== undefined && value !== null
  && (!(Array.isArray(value)) || value.length > 0)
  && (!(typeof value === 'object') || Object.keys(value as Record<string, unknown>).length > 0);

export default function AgentNodeExecutionDetails({ detail }: { detail: AgentRunNodeDetail }) {
  const llm = detail.llm || {};
  const safety = detail.safety || {};
  const eventLlm = detail.event?.llm_result_summary?.llm || {};
  const reasoningAvailable = llm.reasoning_available === true && typeof llm.reasoning === 'string' && llm.reasoning.length > 0;
  const warnings = Array.isArray(detail.event?.warnings) ? detail.event.warnings.map(String) : [];

  return (
    <Stack spacing={0.75} sx={{ minWidth: 0 }}>
      {safety.truncated && <Alert severity="warning">Some invocation data was truncated by trace safety limits.</Alert>}
      {detail.error && <Alert severity="error">{String(detail.error)}</Alert>}
      {(safety.redacted_fields?.length || safety.omitted_fields?.length) && (
        <Alert severity="info">Sensitive or internal fields were removed from this trace.</Alert>
      )}
      {hasData(detail.changes) && <Section title="State changes"><JsonPreview value={detail.changes} maxHeight={320} /></Section>}
      {hasData(detail.checkpoint_before) && <Section title="Checkpoint before" defaultOpen={false}><JsonPreview value={detail.checkpoint_before} maxHeight={440} /></Section>}
      {hasData(detail.checkpoint_after) && <Section title="Checkpoint after" defaultOpen={false}><JsonPreview value={detail.checkpoint_after} maxHeight={440} /></Section>}
      {Array.isArray(llm.prompt) && llm.prompt.length > 0 && <Section title="Prompt" defaultOpen={false}><JsonPreview value={llm.prompt} maxHeight={440} /></Section>}
      {(hasData(llm.response) || hasData(detail.event?.llm_result_summary)) && (
        <Section title="Decision / model output">
          {hasData(llm.response) && <JsonPreview value={llm.response} maxHeight={320} />}
          {hasData(detail.event?.llm_result_summary) && <JsonPreview value={detail.event.llm_result_summary} maxHeight={320} />}
        </Section>
      )}
      {(detail.llm || hasData(detail.event?.llm_result_summary)) && <Section title="Model reasoning" defaultOpen={false}>
        <Typography variant="caption" color="text.secondary">
          Provider-returned reasoning · {llm.reasoning_format || eventLlm.reasoning_format || 'not provided'}
          {eventLlm.token_counts?.reasoning ? ` · ${eventLlm.token_counts.reasoning} reasoning tokens` : ''}
        </Typography>
        {reasoningAvailable ? (
          <Box component="pre" sx={{ m: 0, mt: 0.75, p: 1, maxHeight: 440, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', borderRadius: 1, bgcolor: 'rgba(0,0,0,0.04)', fontSize: '0.72rem' }}>
            {llm.reasoning}
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>Reasoning not provided by model.</Typography>
        )}
      </Section>}
      {Array.isArray(detail.tools) && detail.tools.length > 0 && <Section title="Tools"><JsonPreview value={detail.tools} maxHeight={440} /></Section>}
      {hasData(detail.output) && <Section title="Node output"><JsonPreview value={detail.output} maxHeight={440} /></Section>}
      {warnings.length > 0 && <Section title="Warnings"><Typography variant="caption">{warnings.join(', ')}</Typography></Section>}
      <Divider />
      <Section title="Raw JSON" defaultOpen={false}><JsonPreview value={detail} maxHeight={520} /></Section>
    </Stack>
  );
}
