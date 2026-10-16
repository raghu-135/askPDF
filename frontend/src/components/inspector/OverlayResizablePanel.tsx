import React from 'react';
import { Box, Collapse } from '@mui/material';
import { ResizablePanelHandle } from '../conversation/ResizablePanelHandle';

export function OverlayResizablePanel({
  ratio,
  resizing,
  onResizeStart,
  expanded,
  resizeLabel = 'Resize panel',
  header,
  children,
}: {
  ratio: number;
  resizing: boolean;
  onResizeStart: (event: React.PointerEvent<HTMLDivElement>) => void;
  expanded: boolean;
  resizeLabel?: string;
  header: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Box
      sx={{
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 2,
        height: expanded ? `${ratio * 100}%` : 'auto',
        maxHeight: expanded ? `${ratio * 100}%` : 'none',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        bgcolor: 'background.paper',
        borderTop: 1,
        borderColor: 'divider',
        boxShadow: 3,
        contain: 'layout paint',
        overscrollBehavior: 'contain',
      }}
      onMouseDown={(event) => event.stopPropagation()}
      onWheel={(event) => event.stopPropagation()}
    >
      {expanded ? (
        <ResizablePanelHandle resizing={resizing} onResizeStart={onResizeStart} label={resizeLabel} />
      ) : null}
      <Box sx={{ flexShrink: 0 }}>{header}</Box>
      <Collapse
        in={expanded}
        unmountOnExit
        sx={{
          flex: '1 1 auto',
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          '& .MuiCollapse-wrapper': {
            display: 'flex',
            flex: '1 1 auto',
            minHeight: 0,
          },
          '& .MuiCollapse-wrapperInner': {
            display: 'flex',
            flex: '1 1 auto',
            flexDirection: 'column',
            minHeight: 0,
          },
        }}
      >
        <Box
          sx={{
            flex: '1 1 auto',
            minHeight: 0,
            px: 1.5,
            pb: 1,
            overflow: 'auto',
            overscrollBehavior: 'contain',
            WebkitOverflowScrolling: 'touch',
            touchAction: 'pan-y',
          }}
        >
          {children}
        </Box>
      </Collapse>
    </Box>
  );
}
