import React from 'react';
import { Box, Divider, Tooltip, Typography } from '@mui/material';
import { JsonPreview } from '../inspector/JsonPreview';

const SECTION_HELP: Record<string, string> = {
  Decision: 'Route, execution plan, and LLM decision data produced by router or planner nodes.',
  'Focused Spans': 'Trace spans highlighted for the selected conversation bubble.',
  Input: 'State, refs, and bounded previews available before this node ran.',
  Prompt: 'Rendered LLM prompt summary for nodes that call the model.',
  LLM: 'Model usage, reasoning metadata, token counts, and retry attempts captured for this node.',
  Tools: 'Tool calls made by this worker node, including inputs, results, refs, and warnings.',
  Output: 'State, refs, and bounded previews produced by this node.',
  Warnings: 'Real warning codes from tool/runtime contracts. Skips and planner notes are not warnings.',
  'Raw JSON': 'Normalized trace spans and graph adapter rows used to render this inspector.',
};

export { JsonPreview };

export const hasValue = (value: unknown) => {
  if (!value) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length > 0;
  return true;
};

export const DetailLine = ({ label, value }: { label: string; value?: React.ReactNode }) => {
  if (!value) return null;
  return (
    <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
      <Box component="span" sx={{ fontWeight: 700, color: 'text.primary' }}>{label}: </Box>
      {value}
    </Typography>
  );
};

export const InspectorSection = ({ title, children }: { title: string; children: React.ReactNode }) => {
  if (!children) return null;
  return (
    <>
      <Divider sx={{ my: 1 }} />
      <Tooltip title={SECTION_HELP[title] || ''} placement="top" arrow>
        <Typography variant="caption" sx={{ display: 'inline-block', fontWeight: 700, cursor: 'help' }}>
          {title}
        </Typography>
      </Tooltip>
      {children}
    </>
  );
};

export const TraceObject = ({ value }: { value: unknown }) => (
  hasValue(value) ? <JsonPreview value={value} /> : null
);
