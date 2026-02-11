import React from 'react';
import { truncateFileName } from '../lib/pdf-utils';
import { Box, Tabs, Tab, IconButton, Tooltip, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';

export type PdfTab = {
  id: string;
  fileName: string;
  fileHash: string;
  pdfUrl: string;
  sentences: any[];
  text?: string;
};

type Props = {
  tabs: PdfTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  darkMode?: boolean;
};

export default function PdfTabs({ tabs, activeTabId, onTabChange, onTabClose, darkMode = false }: Props) {
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
              pr: 2.5, // extra space for close button
            },
          }}
        >
          {tabs.map((tab) => (
            <Tab
              key={tab.id}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7 }} />
                  <Tooltip title={tab.fileName} placement="bottom">
                    <Typography
                      component="span"
                      sx={{
                        maxWidth: 120,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {truncateFileName(tab.fileName)}
                    </Typography>
                  </Tooltip>
                  <Tooltip title="Close">
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
                pr: 1.5,
              }}
            />
          ))}
        </Tabs>
        {/* Close buttons are now inside the tab label as a styled span */}
      </Box>
    </Box>
  );
}
