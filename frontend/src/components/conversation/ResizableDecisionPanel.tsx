import React from 'react';
import { Box, IconButton, Typography, useMediaQuery, useTheme } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import { ResizablePanelHandle } from './ResizablePanelHandle';
import { REVIEW_DECISION_PANEL_GRID_ROWS } from '../../lib/deep-research-review-panel-layout';

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
  panelRatio,
  onResizeStart: externalOnResizeStart,
  resizing: externalResizing,
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
  panelRatio?: number;
  onResizeStart?: (event: React.PointerEvent<HTMLDivElement>) => void;
  resizing?: boolean;
}) {
  const theme = useTheme();
  const narrow = useMediaQuery(theme.breakpoints.down('sm'));
  const initialRatio = narrow ? narrowDefaultRatio : defaultRatio;
  const internal = useResizablePanelRatio(rootRef || { current: null }, initialRatio);
  const ratio = panelRatio ?? internal.ratio;
  const resizing = externalResizing ?? internal.resizing;
  const onResizeStart = externalOnResizeStart ?? internal.onResizeStart;

  return (
    <Box sx={{
      display: fillColumn ? 'grid' : 'flex',
      flexDirection: fillColumn ? undefined : 'column',
      gridTemplateRows: fillColumn ? REVIEW_DECISION_PANEL_GRID_ROWS : undefined,
      flex: fillColumn ? '1 1 0' : undefined,
      minWidth: 0,
      mb: 1,
      mx: horizontalInset,
      bgcolor: 'background.default',
      borderRadius: 1,
      maxHeight: fillColumn ? '100%' : `calc(100dvh * ${ratio})`,
      height: fillColumn ? '100%' : undefined,
      minHeight: fillColumn ? 0 : minHeight,
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
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, px: 1, pb: 1, flexShrink: 0, minHeight: 0, overflow: 'hidden' }}>
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
        overflowX: 'hidden',
        minHeight: 0,
      }}>
        {children}
      </Box>
      {footer ? <Box sx={{ flexShrink: 0, px: 1, pb: 1, minHeight: 0 }}>{footer}</Box> : null}
    </Box>
  );
}
