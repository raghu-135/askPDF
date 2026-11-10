import React from 'react';
import { Box, Button } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { REVIEW_TRANSCRIPT_EXPANDED_MAX_HEIGHT } from '../../lib/deep-research-review-panel-layout';

export function DeepResearchReviewTranscriptToggleControl({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <Box
      data-testid="review-transcript-toggle"
      sx={{
        flexShrink: 0,
        borderTop: '1px solid',
        borderColor: 'divider',
        pt: 0.5,
        px: 1,
        bgcolor: 'background.default',
      }}
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
    </Box>
  );
}

/** Transcript body rendered inside `#review-decision-scroll` while review is open. */
export function DeepResearchReviewTranscriptScrollBody({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <Box
      data-testid="review-transcript-body"
      sx={{
        flex: '1 1 0',
        minHeight: 0,
        overflowY: 'auto',
        overflowX: 'hidden',
      }}
    >
      {children}
    </Box>
  );
}

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
    <Box sx={{ flexShrink: 0, borderTop: '1px solid', borderColor: 'divider', pt: 0.5 }}>
      <DeepResearchReviewTranscriptToggleControl open={open} onToggle={onToggle} />
      <Box
        data-testid="review-transcript-body"
        sx={{
          display: open ? 'block' : 'none',
          maxHeight: REVIEW_TRANSCRIPT_EXPANDED_MAX_HEIGHT,
          minHeight: 0,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
