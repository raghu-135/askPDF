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
};

export default function PdfTabs({ tabs, activeTabId, onTabChange, onTabClose }: Props) {
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
        minHeight: 48,
      }}
    >
      <Tabs
        value={currentIndex}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: 48,
          '& .MuiTab-root': {
            minHeight: 48,
            textTransform: 'none',
            fontSize: '0.875rem',
            py: 0,
            pl: 1.5,
            pr: 0.5,
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
                      maxWidth: 150,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {truncateFileName(tab.fileName)}
                  </Typography>
                </Tooltip>
                <Tooltip title="Close">
                  <IconButton
                    size="small"
                    onClick={(e) => handleClose(e, tab.id)}
                    sx={{
                      ml: 0.5,
                      p: 0.25,
                      opacity: 0.6,
                      '&:hover': {
                        opacity: 1,
                        bgcolor: 'action.hover',
                      },
                    }}
                  >
                    <CloseIcon fontSize="small" sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            }
            sx={{
              '&.Mui-selected': {
                bgcolor: 'action.selected',
              },
            }}
          />
        ))}
      </Tabs>
    </Box>
  );
}
