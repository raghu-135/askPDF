import React from 'react';
import { Box, Typography } from '@mui/material';

export default function WorkspaceEmptyState({
  icon,
  title,
  description,
  actions,
  darkMode = false,
  maxWidth = 520,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  actions?: React.ReactNode;
  darkMode?: boolean;
  maxWidth?: number;
}) {
  return (
    <Box
      sx={{
        height: '100%',
        display: 'grid',
        placeItems: 'center',
        bgcolor: darkMode ? '#222' : 'grey.50',
        color: darkMode ? '#eee' : 'inherit',
        p: 4,
      }}
    >
      <Box sx={{ textAlign: 'center', maxWidth }}>
        {icon}
        <Typography variant="h5" gutterBottom sx={{ mt: icon ? 1 : 0 }}>
          {title}
        </Typography>
        {description ? (
          <Typography color="text.secondary" sx={{ mb: actions ? 2.5 : 0 }}>
            {description}
          </Typography>
        ) : null}
        {actions}
      </Box>
    </Box>
  );
}
