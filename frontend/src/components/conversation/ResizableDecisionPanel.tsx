import React from 'react';
import { Box, IconButton, Typography, useMediaQuery, useTheme } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import { ResizablePanelHandle } from './ResizablePanelHandle';
import {
  REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
  REVIEW_DECISION_PANEL_GRID_ROWS,
  REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS,
} from '../../lib/deep-research-review-panel-layout';

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
  showResizeHandle = true,
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
  showResizeHandle?: boolean;
}) {
  const theme = useTheme();
  const narrow = useMediaQuery(theme.breakpoints.down('sm'));
  const initialRatio = narrow ? narrowDefaultRatio : defaultRatio;
  const internal = useResizablePanelRatio(rootRef || { current: null }, initialRatio);
  const ratio = panelRatio ?? internal.ratio;
  const resizing = externalResizing ?? internal.resizing;
  const onResizeStart = externalOnResizeStart ?? internal.onResizeStart;

  const shellBorderSx = {
    borderTop: '1px solid',
    borderColor: variant === 'conflict' ? 'warning.main' : variant === 'approval' ? 'info.main' : 'divider',
  };

  const titleRow = (
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
  );

  const fixedHeaderRow = fixedHeader ? (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      gap: 1,
      px: 1,
      pb: 1,
      flex: '0 1 auto',
      minHeight: 0,
      maxHeight: REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
      overflowY: 'auto',
      overflowX: 'hidden',
    }}>
      {fixedHeader}
    </Box>
  ) : null;

  const bodyRow = (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      gap: 1.5,
      px: 1,
      pt: fixedHeader ? 0 : 1,
      pb: fillColumn && footer ? 0 : (footer ? 0 : 1),
      overflowY: 'auto',
      overflowX: 'hidden',
      minHeight: 0,
      ...(fillColumn ? {} : { flex: '1 1 0' }),
    }}>
      {children}
    </Box>
  );

  const footerRow = footer ? (
    <Box
      data-testid="review-decision-footer"
      sx={{
        flexShrink: 0,
        px: 1,
        pb: 1,
        pt: 0.5,
        borderTop: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      {footer}
    </Box>
  ) : null;

  if (fillColumn) {
    const scrollGridRows = fixedHeader
      ? REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS
      : 'auto auto minmax(0, 1fr)';

    return (
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        flex: '1 1 0',
        minWidth: 0,
        minHeight: 0,
        height: '100%',
        maxHeight: '100%',
        mb: 1,
        mx: horizontalInset,
        bgcolor: 'background.default',
        borderRadius: 1,
        overflow: 'hidden',
        ...shellBorderSx,
        flexShrink: 1,
      }}>
        <Box
          data-testid="review-decision-scroll"
          sx={{
            display: 'grid',
            gridTemplateRows: scrollGridRows,
            flex: '1 1 0',
            minHeight: 0,
            overflowY: 'auto',
            overflowX: 'hidden',
          }}
        >
          {showResizeHandle ? (
            <ResizablePanelHandle resizing={resizing} onResizeStart={onResizeStart} label="Resize decision panel" />
          ) : null}
          {titleRow}
          {fixedHeaderRow}
          {bodyRow}
        </Box>
        {footerRow}
      </Box>
    );
  }

  const shellSx = {
    display: 'flex',
    flexDirection: 'column',
    maxHeight: `calc(100dvh * ${ratio})`,
    minHeight,
  };

  return (
    <Box sx={{
      ...shellSx,
      mb: 1,
      mx: horizontalInset,
      bgcolor: 'background.default',
      borderRadius: 1,
      overflow: 'hidden',
      ...shellBorderSx,
      flexShrink: 0,
    }}>
      {showResizeHandle ? (
        <ResizablePanelHandle resizing={resizing} onResizeStart={onResizeStart} label="Resize decision panel" />
      ) : null}
      {titleRow}
      {fixedHeaderRow}
      {bodyRow}
      {footerRow}
    </Box>
  );
}
