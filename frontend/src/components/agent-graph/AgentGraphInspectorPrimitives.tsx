import React from 'react';
import { Box, Divider, Tooltip, Typography } from '@mui/material';
import { JsonView } from 'react-json-view-lite';

const jsonTreeStyles = {
  container: 'askpdf-json-view',
  childFieldsContainer: 'askpdf-json-view__children',
  basicChildStyle: 'askpdf-json-view__child',
  collapseIcon: 'askpdf-json-view__collapse',
  expandIcon: 'askpdf-json-view__expand',
  collapsedContent: 'askpdf-json-view__collapsed',
  label: 'askpdf-json-view__label',
  clickableLabel: 'askpdf-json-view__clickable-label',
  nullValue: 'askpdf-json-view__null',
  undefinedValue: 'askpdf-json-view__undefined',
  numberValue: 'askpdf-json-view__number',
  stringValue: 'askpdf-json-view__string',
  booleanValue: 'askpdf-json-view__boolean',
  otherValue: 'askpdf-json-view__other',
  punctuation: 'askpdf-json-view__punctuation',
  quotesForFieldNames: false,
  stringifyStringValues: true,
  ariaLables: {
    collapseJson: 'Collapse JSON node',
    expandJson: 'Expand JSON node',
  },
};

const SECTION_HELP: Record<string, string> = {
  Decision: 'Route, execution plan, and LLM decision data produced by router or planner nodes.',
  Input: 'State, refs, and bounded previews available before this node ran.',
  Prompt: 'Rendered LLM prompt summary for nodes that call the model.',
  Tools: 'Tool calls made by this worker node, including inputs, results, refs, and warnings.',
  Output: 'State, refs, and bounded previews produced by this node.',
  Warnings: 'Real warning codes from tool/runtime contracts. Skips and planner notes are not warnings.',
  'Raw JSON': 'Normalized trace spans and graph adapter rows used to render this inspector.',
};

const shouldExpandJsonNode = (level: number) => level < 4;

const isJsonViewData = (value: unknown): value is Record<string, unknown> | unknown[] => {
  return value !== null && typeof value === 'object';
};

export const hasValue = (value: unknown) => {
  if (!value) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length > 0;
  return true;
};

export const JsonPreview = ({ value, maxHeight = 140 }: { value: unknown; maxHeight?: number }) => {
  const jsonSx = {
    '& .askpdf-json-view': {
      lineHeight: 1.35,
      whiteSpace: 'pre-wrap',
      overflowWrap: 'break-word',
      color: 'text.primary',
    },
    '& .askpdf-json-view__children': {
      m: 0,
      pl: 1.5,
      listStyle: 'none',
      borderLeft: '1px solid',
      borderColor: 'divider',
    },
    '& .askpdf-json-view__child': {
      m: 0,
      py: 0.1,
    },
    '& .askpdf-json-view__collapse, & .askpdf-json-view__expand': {
      display: 'inline-block',
      width: 14,
      mr: 0.5,
      cursor: 'pointer',
      userSelect: 'none',
      color: 'text.secondary',
    },
    '& .askpdf-json-view__collapse::after': {
      content: '"▾"',
    },
    '& .askpdf-json-view__expand::after': {
      content: '"▸"',
    },
    '& .askpdf-json-view__collapsed': {
      color: 'text.disabled',
      fontStyle: 'italic',
      mx: 0.5,
    },
    '& .askpdf-json-view__collapsed::after': {
      content: '"..."',
    },
    '& .askpdf-json-view__label, & .askpdf-json-view__clickable-label': {
      color: 'primary.main',
      fontWeight: 700,
      mr: 0.5,
    },
    '& .askpdf-json-view__clickable-label': {
      cursor: 'pointer',
    },
    '& .askpdf-json-view__string': {
      color: 'success.dark',
    },
    '& .askpdf-json-view__number': {
      color: 'secondary.main',
    },
    '& .askpdf-json-view__boolean': {
      color: 'warning.dark',
    },
    '& .askpdf-json-view__null, & .askpdf-json-view__undefined': {
      color: 'text.disabled',
      fontStyle: 'italic',
    },
    '& .askpdf-json-view__punctuation, & .askpdf-json-view__other': {
      color: 'text.secondary',
    },
  };

  return (
    <Box
      sx={{
        m: 0,
        mt: 0.5,
        p: 0.75,
        maxHeight,
        overflow: 'auto',
        borderRadius: 1,
        bgcolor: 'rgba(0,0,0,0.04)',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
        fontSize: '0.68rem',
        ...jsonSx,
      }}
    >
      {isJsonViewData(value) ? (
        <JsonView
          data={value}
          style={jsonTreeStyles}
          shouldExpandNode={shouldExpandJsonNode}
          clickToExpandNode
          compactTopLevel
        />
      ) : (
        <Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {JSON.stringify(value, null, 2)}
        </Box>
      )}
    </Box>
  );
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
