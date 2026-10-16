import React from 'react';
import { Typography, type SxProps, type Theme } from '@mui/material';

const MONO_FONT = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';

export function InspectorTextPreview({
  text,
  maxHeight,
  sx,
}: {
  text: string;
  maxHeight?: number | false;
  sx?: SxProps<Theme>;
}) {
  return (
    <Typography
      component="pre"
      variant="caption"
      sx={{
        display: 'block',
        m: 0,
        p: 1,
        borderRadius: 1,
        bgcolor: 'action.hover',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: MONO_FONT,
        ...(maxHeight !== false && maxHeight != null ? { maxHeight, overflow: 'auto' } : {}),
        ...sx,
      }}
    >
      {text || '(empty text)'}
    </Typography>
  );
}
