import React from 'react';
import { truncateFileName } from '../lib/pdf-utils';
import { Box, Tabs, Tab, IconButton, Tooltip, Typography, CircularProgress } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import CloseIcon from '@mui/icons-material/Close';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import LanguageIcon from '@mui/icons-material/Language';
import OpenInBrowserIcon from '@mui/icons-material/OpenInBrowser';
import AddIcon from '@mui/icons-material/Add';
import type { BackendSentence, BBox } from '../lib/bbox-derivation';

type Sentence = Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] };

export type PdfTab = {
  id: string;
  fileName: string;
  fileHash: string;
  pdfUrl: string;
  sentences: Sentence[] | null;
  text?: string;
  sourceType?: 'pdf' | 'browser';
  sourceUrl?: string;
  parsingStatus?: 'pending' | 'completed' | 'failed';
};

type Props = {
  tabs: PdfTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  /** When provided, shows a remove-from-thread trash icon on each tab. */
  onTabRemove?: (tabId: string) => void;
  darkMode?: boolean;
  /** Whether to show the browser tab */
  showBrowserTab?: boolean;
  /** Callback when browser tab is clicked */
  onBrowserTabClick?: () => void;
  /** Callback when add-to-thread button is clicked on browser tab */
  onAddBrowserToThread?: () => void;
  /** Whether browser capture is in progress */
  isBrowserCapturing?: boolean;
};

const PdfTabs = React.memo(function PdfTabs({ tabs, activeTabId, onTabChange, onTabClose, onTabRemove, darkMode = false, showBrowserTab = false, onBrowserTabClick, onAddBrowserToThread, isBrowserCapturing = false }: Props) {
  const browserTabId = 'browser-tab';

  if (tabs.length === 0 && !showBrowserTab) {
    return null;
  }

  const activeIndex = tabs.findIndex(t => t.id === activeTabId);
  const currentIndex = activeIndex >= 0 ? activeIndex : 0;
  const isBrowserActive = activeTabId === browserTabId;
  const displayIndex = isBrowserActive ? 0 : (showBrowserTab ? currentIndex + 1 : currentIndex);

  const handleTabChange = (_event: React.SyntheticEvent, newIndex: number) => {
    // Handle browser tab click
    if (showBrowserTab && newIndex === 0) {
      onBrowserTabClick?.();
      return;
    }
    // Adjust index for regular tabs (browser tab is at index 0)
    const adjustedIndex = showBrowserTab ? newIndex - 1 : newIndex;
    if (tabs[adjustedIndex]) {
      onTabChange(tabs[adjustedIndex].id);
    }
  };

  const handleClose = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    onTabClose(tabId);
  };

  const handleRemove = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    onTabRemove?.(tabId);
  };

  return (
    <Box
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        minHeight: 40,
        color: 'text.primary',
      }}
    >
      <Box sx={{ position: 'relative' }}>
        <Tabs
          value={displayIndex}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            minHeight: 40,
            '& .MuiTab-root': {
              minHeight: 40,
              textTransform: 'none',
              fontSize: '0.875rem',
              py: 0,
              pl: 1.5,
              pr: 2.5,
            },
          }}
        >
          {showBrowserTab && (
            <Tab
              key={browserTabId}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <OpenInBrowserIcon fontSize="small" />
                  <Typography component="span">Browser</Typography>
                  {/* Add to Thread button - only when browser tab is active */}
                  {isBrowserActive && onAddBrowserToThread && (
                    <Tooltip title="Add current page to thread">
                      <span>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            onAddBrowserToThread();
                          }}
                          disabled={isBrowserCapturing}
                          sx={{ ml: 0.5, p: 0.3 }}
                        >
                          {isBrowserCapturing ? (
                            <CircularProgress size={14} />
                          ) : (
                            <AddIcon fontSize="small" />
                          )}
                        </IconButton>
                      </span>
                    </Tooltip>
                  )}
                </Box>
              }
              sx={{
                '&.Mui-selected': {
                  bgcolor: 'action.selected',
                },
                pr: 1.5,
              }}
            />
          )}
          {tabs.map((tab) => {
            const isBrowser = tab.sourceType === 'browser';
            const label = isBrowser
              ? (tab.sourceUrl ? new URL(tab.sourceUrl).hostname : tab.fileName)
              : truncateFileName(tab.fileName);
            const fullTitle = isBrowser ? (tab.sourceUrl || tab.fileName) : tab.fileName;

            return (
              <Tab
                key={tab.id}
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {isBrowser
                      ? (
                        <Tooltip title="Open source webpage">
                          <span
                            onClick={(e) => {
                              e.stopPropagation();
                              if (tab.sourceUrl) {
                                window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                              }
                            }}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              cursor: 'pointer',
                              padding: '2px',
                              borderRadius: '4px',
                              color: '#1976d2',
                            }}
                          >
                            <LanguageIcon fontSize="small" />
                          </span>
                        </Tooltip>
                      )
                      : <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7 }} />
                    }
                    <Tooltip title={fullTitle} placement="bottom">
                      <Typography
                        component="span"
                        sx={{
                          maxWidth: 120,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {label}
                      </Typography>
                    </Tooltip>

                    {/* Remove from thread (trash) — only when callback provided */}
                    {onTabRemove && (
                      <Tooltip title="Remove from thread">
                        <span
                          onClick={(e) => handleRemove(e, tab.id)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            marginLeft: 2,
                            cursor: 'pointer',
                            opacity: 0,
                            borderRadius: 4,
                            padding: 2,
                            transition: 'opacity 0.2s',
                          }}
                          className="tab-remove-btn"
                          onMouseOver={e => (e.currentTarget.style.opacity = '1')}
                          onMouseOut={e => (e.currentTarget.style.opacity = '0')}
                        >
                          <DeleteIcon fontSize="small" sx={{ fontSize: 14, color: 'error.main' }} />
                        </span>
                      </Tooltip>
                    )}

                    {/* Close tab (X) */}
                    <Tooltip title="Close tab">
                      <span
                        onClick={(e) => {
                          e.stopPropagation();
                          handleClose(e, tab.id);
                        }}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          marginLeft: 4,
                          cursor: 'pointer',
                          opacity: 0.6,
                          borderRadius: 4,
                          padding: 2,
                          transition: 'opacity 0.2s',
                        }}
                        onMouseOver={e => (e.currentTarget.style.opacity = '1')}
                        onMouseOut={e => (e.currentTarget.style.opacity = '0.6')}
                      >
                        <CloseIcon fontSize="small" sx={{ fontSize: 16 }} />
                      </span>
                    </Tooltip>
                  </Box>
                }
                sx={{
                  '&.Mui-selected': {
                    bgcolor: 'action.selected',
                  },
                  '&:hover .tab-remove-btn': {
                    opacity: '0.7 !important',
                  },
                  pr: 1.5,
                }}
              />
            );
          })}
        </Tabs>
      </Box>
    </Box>
  );
});

export default PdfTabs;
