import React from 'react';
import { Box } from '@mui/material';

export function ResizablePanelHandle({
  resizing,
  onResizeStart,
  label = 'Resize panel',
}: {
  resizing: boolean;
  onResizeStart: (event: React.PointerEvent<HTMLDivElement>) => void;
  label?: string;
}) {
  return (
    <Box
      onPointerDown={onResizeStart}
      role="separator"
      aria-orientation="horizontal"
      aria-label={label}
      sx={{
        flex: '0 0 auto',
        height: 24,
        cursor: 'ns-resize',
        touchAction: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'text.secondary',
        '&::before': {
          content: '""',
          width: '18%',
          minWidth: 32,
          maxWidth: 80,
          height: 4,
          borderRadius: 999,
          bgcolor: resizing ? 'primary.main' : 'divider',
        },
        '&:hover::before': { bgcolor: 'primary.main' },
      }}
    />
  );
}
