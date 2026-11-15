import React from 'react';
import { Box } from '@mui/material';

export function ResizablePanelHandle({
  resizing,
  onResizeStart,
  onKeyboardResize,
  label = 'Resize panel',
}: {
  resizing: boolean;
  onResizeStart: (event: React.PointerEvent<HTMLDivElement>) => void;
  onKeyboardResize?: (deltaRatio: number) => void;
  label?: string;
}) {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (!onKeyboardResize) return;
    const step = event.shiftKey ? 0.08 : 0.04;
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      onKeyboardResize(step);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      onKeyboardResize(-step);
    }
  };

  return (
    <Box
      onPointerDown={onResizeStart}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="separator"
      aria-orientation="horizontal"
      aria-label={label}
      aria-valuetext={resizing ? 'Resizing' : undefined}
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
