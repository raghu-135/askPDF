import React from 'react';
import { Box, CircularProgress, Tooltip, Typography } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';

export default function EmbeddingModelReadinessIndicator({
  model,
  ready,
  size = 20,
  showStatusLabel = false,
}: {
  model: string;
  ready: boolean | null;
  size?: number;
  showStatusLabel?: boolean;
}) {
  const status = ready === null ? 'checking' : ready ? 'ready' : 'unavailable';
  return (
    <Tooltip title={model ? `${model}: ${status}` : `Embedding model: ${status}`}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flex: '0 0 auto' }}>
        {ready === null
          ? <CircularProgress size={size} />
          : ready
            ? <CheckCircleIcon color="success" sx={{ fontSize: size }} />
            : <ErrorIcon color="error" sx={{ fontSize: size }} />}
        {showStatusLabel && ready !== true && (
          <Typography
            variant="caption"
            color={ready === false ? 'error' : 'warning.main'}
            sx={{ fontWeight: 700 }}
          >
            {ready === false ? 'OFFLINE' : 'CHECKING...'}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
}
