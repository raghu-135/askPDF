import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Box, IconButton, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { clampDecisionPanelRatio } from '../../lib/conversation-ui-state';

export function ResizableDecisionPanel({
  title,
  children,
  variant = 'clarification',
  onClose,
  rootRef,
  defaultRatio = 0.3,
  horizontalInset = 0,
  minHeight = 0,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
  variant?: 'clarification' | 'conflict' | 'approval';
  onClose?: () => void;
  rootRef?: React.RefObject<HTMLElement | null>;
  defaultRatio?: number;
  horizontalInset?: number;
  minHeight?: number;
}) {
  const [ratio, setRatio] = useState(() => clampDecisionPanelRatio(defaultRatio));
  const [resizing, setResizing] = useState(false);
  const resizeRef = useRef({ startY: 0, startRatio: ratio });

  const handleResizeStart = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    resizeRef.current = { startY: event.clientY, startRatio: ratio };
    setResizing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [ratio]);

  const handleResizeMove = useCallback((event: PointerEvent) => {
    const panelHeight = rootRef?.current?.getBoundingClientRect().height || window.innerHeight;
    const deltaRatio = (resizeRef.current.startY - event.clientY) / panelHeight;
    setRatio(clampDecisionPanelRatio(resizeRef.current.startRatio + deltaRatio));
  }, [rootRef]);

  const handleResizeEnd = useCallback(() => setResizing(false), []);

  useEffect(() => {
    if (!resizing) return;
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('pointermove', handleResizeMove);
    document.addEventListener('pointerup', handleResizeEnd);
    document.addEventListener('pointercancel', handleResizeEnd);
    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('pointermove', handleResizeMove);
      document.removeEventListener('pointerup', handleResizeEnd);
      document.removeEventListener('pointercancel', handleResizeEnd);
    };
  }, [handleResizeEnd, handleResizeMove, resizing]);

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      mb: 1,
      mx: horizontalInset,
      bgcolor: 'background.default',
      borderRadius: 1,
      maxHeight: `calc(100dvh * ${ratio})`,
      minHeight,
      overflow: 'hidden',
      borderTop: '1px solid',
      borderColor: variant === 'conflict' ? 'warning.main' : variant === 'approval' ? 'info.main' : 'divider',
      flexShrink: 0,
    }}>
      <Box
        onPointerDown={handleResizeStart}
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize decision panel"
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
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1, pb: 1, flexShrink: 0 }}>
        <Typography variant="caption" sx={{ flex: 1, textAlign: 'center', color: 'text.secondary', fontWeight: 'bold' }}>
          {title}
        </Typography>
        {onClose && (
          <IconButton size="small" onClick={onClose} aria-label="Close decision panel" sx={{ flex: '0 0 auto' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, px: 1, pt: 1, pb: 1, overflowY: 'auto', minHeight: 0 }}>
        {children}
      </Box>
    </Box>
  );
}
