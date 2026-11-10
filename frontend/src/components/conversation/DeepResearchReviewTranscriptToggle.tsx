import React from 'react';
import { Box, Button, Collapse } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

export function DeepResearchReviewTranscriptToggle({
  open,
  onToggle,
  children,
}: {
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <Box
      data-testid="review-transcript-toggle"
      sx={{ flexShrink: 0, borderTop: '1px solid', borderColor: 'divider', pt: 0.5 }}
    >
      <Button
        size="small"
        color="inherit"
        onClick={onToggle}
        aria-expanded={open}
        data-testid="review-transcript-toggle-button"
        startIcon={open ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        sx={{ justifyContent: 'flex-start', textTransform: 'none', width: '100%', minHeight: 32, py: 0.25 }}
      >
        {open ? 'Hide transcript' : 'Show transcript'}
      </Button>
      <Collapse in={open} unmountOnExit={false}>
        <Box
          data-testid="review-transcript-body"
          sx={{ maxHeight: 'min(42vh, 360px)', minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}
        >
          {children}
        </Box>
      </Collapse>
    </Box>
  );
}
