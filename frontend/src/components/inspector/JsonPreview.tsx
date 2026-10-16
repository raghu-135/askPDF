import React from 'react';
import { Box } from '@mui/material';
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

const shouldExpandJsonNode = (level: number) => level < 2;

const isJsonViewData = (value: unknown): value is Record<string, unknown> | unknown[] => {
  return value !== null && typeof value === 'object';
};

export const JsonPreview = React.memo(function JsonPreview({
  value,
  maxHeight = 140,
}: {
  value: unknown;
  maxHeight?: number | false;
}) {
  const jsonSx = {
    '& .askpdf-json-view': {
      lineHeight: 1.35,
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere',
      wordBreak: 'break-word',
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
      overflowWrap: 'anywhere',
      wordBreak: 'break-word',
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
        ...(maxHeight !== false ? { maxHeight, overflow: 'auto' } : {}),
        minWidth: 0,
        maxWidth: '100%',
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
});
