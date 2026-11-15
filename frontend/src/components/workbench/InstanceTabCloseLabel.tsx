import React from 'react';
import { Box, IconButton, Tooltip, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export default function InstanceTabCloseLabel({
  title,
  adornment,
  onClose,
  closeTooltip = 'Close tab',
}: {
  title: string;
  adornment?: React.ReactNode;
  onClose: () => void;
  closeTooltip?: string;
}) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
      <Typography variant="body2" noWrap sx={{ maxWidth: 170 }}>
        {title}
      </Typography>
      {adornment}
      <Tooltip title={closeTooltip}>
        <IconButton
          size="small"
          onClick={(event) => {
            event.stopPropagation();
            onClose();
          }}
          sx={{ p: 0.2 }}
        >
          <CloseIcon sx={{ fontSize: 15 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
}