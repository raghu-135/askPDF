import React from 'react';
import { truncateFileName } from '../lib/pdf-utils';
import { Box, Tabs, Tab, IconButton, Tooltip, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import LanguageIcon from '@mui/icons-material/Language';

export type PdfTab = {
  id: string;
  fileName: string;
  fileHash: string;
  pdfUrl: string;
  sentences: any[];
  text?: string;
  sourceType?: 'pdf' | 'web';
  sourceUrl?: string;
};

type Props = {
  tabs: PdfTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  /** When provided, shows a remove-from-thread trash icon on each tab. */
  onTabRemove?: (tabId: string) => void;
  darkMode?: boolean;
};

export default function PdfTabs({ tabs, activeTabId, onTabChange, onTabClose, onTabRemove, darkMode = false }: Props) {
  if (tabs.length === 0) {
    return null;
  }

  const activeIndex = tabs.findIndex(t => t.id === activeTabId);
  const currentIndex = activeIndex >= 0 ? activeIndex : 0;

  const handleTabChange = (_event: React.SyntheticEvent, newIndex: number) => {
    if (tabs[newIndex]) {
      onTabChange(tabs[newIndex].id);
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
          value={currentIndex}
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
          {tabs.map((tab) => {
            const isWeb = tab.sourceType === 'web';
            const label = isWeb
              ? (tab.sourceUrl ? new URL(tab.sourceUrl).hostname : tab.fileName)
              : truncateFileName(tab.fileName);
            const fullTitle = isWeb ? (tab.sourceUrl || tab.fileName) : tab.fileName;

            return (
              <Tab
                key={tab.id}
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {isWeb
                      ? <LanguageIcon fontSize="small" sx={{ color: 'primary.main', opacity: 0.8 }} />
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
}
