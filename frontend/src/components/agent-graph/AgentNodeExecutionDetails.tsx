import React from 'react';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { Alert, Box, Divider, Stack, Typography } from '@mui/material';
import type { AgentRunOperationDetail } from '../../lib/api';
import { ConversationDisclosure } from '../conversation/ConversationDisclosure';
import { JsonPreview } from '../inspector/JsonPreview';

const hasData = (value: unknown) => value !== undefined && value !== null
  && (!(Array.isArray(value)) || value.length > 0)
  && (!(typeof value === 'object') || Object.keys(value as Record<string, unknown>).length > 0);

function AgentNodeExecutionDetails({ detail }: { detail: AgentRunOperationDetail }) {
  const llm = detail.llm || {};
  const safety = detail.safety || {};
  const eventLlm = detail.event?.llm_result_summary?.llm || {};
  const reasoningAvailable = llm.reasoning_available === true && typeof llm.reasoning === 'string' && llm.reasoning.length > 0;
  const warnings = Array.isArray(detail.event?.warnings) ? detail.event.warnings.map(String) : [];
  const memoryRefs = detail.event?.output_refs?.memories
    || detail.event?.artifact_refs?.memories
    || detail.output?.memory_refs
    || detail.output?.refs?.memories;
  const memoryToolEvent = Array.isArray(detail.output?.tool_events)
    ? detail.output.tool_events.find((event: Record<string, any>) => event?.tool_name === 'search_long_term_memory')
    : undefined;
  const memoryScopes = detail.output?.memory_scopes
    || detail.output?.artifacts?.memory_scopes
    || detail.event?.artifact_refs?.memory_scopes
    || memoryToolEvent?.artifacts?.memory_scopes;
  const memoryScopePolicy = detail.output?.memory_scope_policy
    || detail.output?.artifacts?.memory_scope_policy
    || detail.event?.artifact_refs?.memory_scope_policy
    || memoryToolEvent?.artifacts?.memory_scope_policy;
  const memoryAppliedOverrides = detail.output?.artifacts?.memory_applied_overrides
    || memoryToolEvent?.artifacts?.memory_applied_overrides;
  const memorySuppressedIds = detail.output?.artifacts?.memory_suppressed_ids
    || memoryToolEvent?.artifacts?.memory_suppressed_ids;
  const errorText = typeof detail.error === 'string'
    ? detail.error
    : detail.error && typeof detail.error === 'object'
      ? String((detail.error as Record<string, any>).raw_message || (detail.error as Record<string, any>).message || JSON.stringify(detail.error))
      : '';

  return (
    <Stack spacing={0.75} sx={{ minWidth: 0 }}>
      {safety.truncated && (
        <Stack direction="row" spacing={0.5} alignItems="flex-start" sx={{ color: 'warning.main', minWidth: 0 }}>
          <WarningAmberIcon sx={{ fontSize: 15, mt: '1px', flexShrink: 0 }} />
          <Typography variant="caption" sx={{ minWidth: 0, overflowWrap: 'anywhere' }}>Some invocation data was truncated by trace safety limits.</Typography>
        </Stack>
      )}
      {errorText && <Alert severity="error" sx={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}>{errorText}</Alert>}
      {(safety.redacted_fields?.length || safety.omitted_fields?.length) && (
        <Stack direction="row" spacing={0.5} alignItems="flex-start" sx={{ color: 'info.main', minWidth: 0 }}>
          <InfoOutlinedIcon sx={{ fontSize: 15, mt: '1px', flexShrink: 0 }} />
          <Typography variant="caption" sx={{ minWidth: 0, overflowWrap: 'anywhere' }}>Sensitive or internal fields were removed from this trace.</Typography>
        </Stack>
      )}
      {hasData(detail.changes) && (
        <ConversationDisclosure label="State changes" defaultExpanded>
          <JsonPreview value={detail.changes} maxHeight={320} />
        </ConversationDisclosure>
      )}
      {hasData(detail.checkpoint_before) && (
        <ConversationDisclosure label="Checkpoint before">
          <JsonPreview value={detail.checkpoint_before} maxHeight={440} />
        </ConversationDisclosure>
      )}
      {hasData(detail.checkpoint_after) && (
        <ConversationDisclosure label="Checkpoint after">
          <JsonPreview value={detail.checkpoint_after} maxHeight={440} />
        </ConversationDisclosure>
      )}
      {Array.isArray(llm.prompt) && llm.prompt.length > 0 && (
        <ConversationDisclosure label="Prompt">
          <JsonPreview value={llm.prompt} maxHeight={440} />
        </ConversationDisclosure>
      )}
      {(hasData(llm.response) || hasData(detail.event?.llm_result_summary)) && (
        <ConversationDisclosure label="Decision / model output" defaultExpanded>
          {hasData(llm.response) && <JsonPreview value={llm.response} maxHeight={320} />}
          {hasData(detail.event?.llm_result_summary) && <JsonPreview value={detail.event.llm_result_summary} maxHeight={320} />}
        </ConversationDisclosure>
      )}
      {(detail.llm || hasData(detail.event?.llm_result_summary)) && (
        <ConversationDisclosure label="Model reasoning">
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
        </ConversationDisclosure>
      )}
      {Array.isArray(detail.tools) && detail.tools.length > 0 && (
        <ConversationDisclosure label="Tools" defaultExpanded>
          <JsonPreview value={detail.tools} maxHeight={440} />
        </ConversationDisclosure>
      )}
      {(hasData(memoryRefs) || hasData(memoryScopes) || hasData(memoryScopePolicy) || hasData(memoryAppliedOverrides) || hasData(memorySuppressedIds)) && (
        <ConversationDisclosure label="Memory refs" defaultExpanded>
          {hasData(memoryRefs) && <JsonPreview value={{ memories: memoryRefs }} maxHeight={220} />}
          {hasData(memoryScopes) && <JsonPreview value={{ scopes: memoryScopes }} maxHeight={220} />}
          {hasData(memoryScopePolicy) && <JsonPreview value={{ policy: memoryScopePolicy }} maxHeight={220} />}
          {hasData(memoryAppliedOverrides) && <JsonPreview value={{ applied_overrides: memoryAppliedOverrides }} maxHeight={220} />}
          {hasData(memorySuppressedIds) && <JsonPreview value={{ suppressed_memory_ids: memorySuppressedIds }} maxHeight={220} />}
        </ConversationDisclosure>
      )}
      {hasData(detail.output) && (
        <ConversationDisclosure label="Node output" defaultExpanded>
          <JsonPreview value={detail.output} maxHeight={440} />
        </ConversationDisclosure>
      )}
      {warnings.length > 0 && (
        <ConversationDisclosure label="Warnings" defaultExpanded>
          <Typography variant="caption" sx={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}>{warnings.join(', ')}</Typography>
        </ConversationDisclosure>
      )}
      <Divider />
      <ConversationDisclosure label="Raw JSON">
        <JsonPreview value={detail} maxHeight={520} />
      </ConversationDisclosure>
    </Stack>
  );
}

export default React.memo(AgentNodeExecutionDetails);
