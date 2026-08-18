import React from 'react';
import { Box, type SxProps, type Theme } from '@mui/material';

export function WorkbenchToolbarTrailingActions({
  children,
  sx,
}: {
  children: React.ReactNode;
  sx?: SxProps<Theme>;
}) {
  return (
    <Box
      sx={[
        {
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          ml: 'auto',
          flex: '0 0 auto',
        },
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
    >
      {children}
    </Box>
  );
}
