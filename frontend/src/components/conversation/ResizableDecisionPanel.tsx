import React from 'react';
import { Box, IconButton, Typography, useMediaQuery, useTheme } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import { ResizablePanelHandle } from './ResizablePanelHandle';
import {
  REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
  REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT,
  REVIEW_DECISION_PANEL_GRID_ROWS,
  REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS,
} from '../../lib/deep-research-review-panel-layout';

export function ResizableDecisionPanel({
  title,
  children = null,
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
  betweenScrollAndFooter,
}: {
  title: React.ReactNode;
  children?: React.ReactNode;
  footer?: React.ReactNode;
  fixedHeader?: React.ReactNode;
  betweenScrollAndFooter?: React.ReactNode;
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
      minHeight: 0,
      overflow: 'hidden',
      ...(fillColumn
        ? { flex: '1 1 0', minHeight: 0 }
        : {
          height: '100%',
          maxHeight: REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
          overflowY: 'auto',
          overflowX: 'hidden',
        }),
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
      overflowX: 'hidden',
      minHeight: 0,
      ...(fillColumn ? { flexShrink: 0, overflowY: 'visible' } : { flex: '1 1 0', overflowY: 'auto' }),
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
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        maxHeight: REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT,
        overflowY: 'auto',
        '& > :last-child': { flexShrink: 0 },
      }}
    >
      {footer}
    </Box>
  ) : null;

  if (fillColumn) {
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
        {showResizeHandle ? (
          <ResizablePanelHandle resizing={resizing} onResizeStart={onResizeStart} label="Resize decision panel" />
        ) : null}
        {titleRow}
        <Box
          data-testid="review-decision-scroll"
          sx={{
            flex: '1 1 0',
            minHeight: 0,
            overflowX: 'hidden',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            px: 1,
            pb: 1,
            position: 'relative',
            zIndex: 0,
            isolation: 'isolate',
          }}
        >
          {fixedHeader}
          {children}
        </Box>
        {betweenScrollAndFooter ? (
          <Box sx={{ flexShrink: 0, position: 'relative', zIndex: 1, bgcolor: 'background.default' }}>
            {betweenScrollAndFooter}
          </Box>
        ) : null}
        {footer ? (
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
              display: 'flex',
              flexDirection: 'column',
              gap: 0.5,
              maxHeight: REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT,
              overflow: 'hidden',
              position: 'relative',
              zIndex: 2,
              boxShadow: '0 -4px 12px rgba(0,0,0,0.06)',
            }}
          >
            {footer}
          </Box>
        ) : null}
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
