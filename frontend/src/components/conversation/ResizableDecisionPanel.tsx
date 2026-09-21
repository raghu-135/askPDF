import React from 'react';
import { Box, IconButton, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import { ResizablePanelHandle } from './ResizablePanelHandle';

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
  const { ratio, resizing, onResizeStart } = useResizablePanelRatio(rootRef || { current: null }, defaultRatio);

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
      <ResizablePanelHandle resizing={resizing} onResizeStart={onResizeStart} label="Resize decision panel" />
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
