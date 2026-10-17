import React from 'react';
import { Box, IconButton, Typography, useMediaQuery, useTheme } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import { ResizablePanelHandle } from './ResizablePanelHandle';

export function ResizableDecisionPanel({
  title,
  children,
  footer,
  fixedHeader,
  variant = 'clarification',
  onClose,
  rootRef,
  defaultRatio = 0.3,
  narrowDefaultRatio = 0.32,
  horizontalInset = 0,
  minHeight = 0,
  fillColumn = false,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  fixedHeader?: React.ReactNode;
  variant?: 'clarification' | 'conflict' | 'approval';
  onClose?: () => void;
  rootRef?: React.RefObject<HTMLElement | null>;
  defaultRatio?: number;
  narrowDefaultRatio?: number;
  horizontalInset?: number;
  minHeight?: number;
  fillColumn?: boolean;
}) {
  const theme = useTheme();
  const narrow = useMediaQuery(theme.breakpoints.down('sm'));
  const initialRatio = narrow ? narrowDefaultRatio : defaultRatio;
  const { ratio, resizing, onResizeStart } = useResizablePanelRatio(rootRef || { current: null }, initialRatio);

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      flex: fillColumn ? '1 1 0' : undefined,
      mb: 1,
      mx: horizontalInset,
      bgcolor: 'background.default',
      borderRadius: 1,
      maxHeight: fillColumn ? '100%' : `calc(100dvh * ${ratio})`,
      height: fillColumn ? '100%' : undefined,
      minHeight,
      overflow: 'hidden',
      borderTop: '1px solid',
      borderColor: variant === 'conflict' ? 'warning.main' : variant === 'approval' ? 'info.main' : 'divider',
      flexShrink: fillColumn ? 1 : 0,
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
      {fixedHeader ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, px: 1, pb: 1, flexShrink: 0 }}>
          {fixedHeader}
        </Box>
      ) : null}
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        px: 1,
        pt: fixedHeader ? 0 : 1,
        pb: footer ? 0 : 1,
        overflowY: 'auto',
        minHeight: 0,
        flex: 1,
      }}>
        {children}
      </Box>
      {footer ? <Box sx={{ flexShrink: 0, px: 1, pb: 1 }}>{footer}</Box> : null}
    </Box>
  );
}
