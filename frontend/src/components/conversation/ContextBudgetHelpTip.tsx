import React from 'react';
import { IconButton, Tooltip, Typography } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

const tooltipSlotProps = {
  tooltip: { sx: { maxWidth: 300 } },
};

export function ContextBudgetHelpTip({
  text,
  ariaLabel = 'Help',
}: {
  text: React.ReactNode;
  ariaLabel?: string;
}) {
  const title = typeof text === 'string' ? (
    <Typography variant="caption" sx={{ display: 'block', whiteSpace: 'pre-wrap' }}>
      {text}
    </Typography>
  ) : text;

  return (
    <Tooltip title={title} placement="top" enterDelay={300} slotProps={tooltipSlotProps}>
      <IconButton
        size="small"
        aria-label={ariaLabel}
        onClick={(event) => event.stopPropagation()}
        sx={{ p: 0.2, color: 'text.secondary', verticalAlign: 'middle' }}
      >
        <InfoOutlinedIcon sx={{ fontSize: 15 }} />
      </IconButton>
    </Tooltip>
  );
}
