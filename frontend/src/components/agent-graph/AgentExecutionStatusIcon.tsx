import React from 'react';
import CancelIcon from '@mui/icons-material/Cancel';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PauseCircleIcon from '@mui/icons-material/PauseCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import RemoveCircleIcon from '@mui/icons-material/RemoveCircle';
import ScheduleIcon from '@mui/icons-material/Schedule';
import { Box, CircularProgress, Tooltip } from '@mui/material';
import { getAgentExecutionStatusPresentation } from './agent-execution-status';

const paletteColor = {
  success: 'success.main',
  error: 'error.main',
  warning: 'warning.main',
  primary: 'primary.main',
  disabled: 'text.disabled',
} as const;

export default function AgentExecutionStatusIcon({ status, size = 16, showTooltip = true }: { status?: unknown; size?: number; showTooltip?: boolean }) {
  const presentation = getAgentExecutionStatusPresentation(status);
  const sx = { fontSize: size, color: paletteColor[presentation.color] };
  const icon = presentation.icon === 'spinner'
    ? <CircularProgress size={size - 1} thickness={5} color="primary" />
    : presentation.icon === 'check'
      ? <CheckCircleIcon sx={sx} />
      : presentation.icon === 'cross'
        ? <CancelIcon sx={sx} />
        : presentation.icon === 'minus'
          ? <RemoveCircleIcon sx={sx} />
          : presentation.icon === 'pause'
            ? <PauseCircleIcon sx={sx} />
            : presentation.icon === 'clock'
              ? <ScheduleIcon sx={sx} />
              : <RadioButtonUncheckedIcon sx={sx} />;

  const content = (
    <Box component="span" role="img" aria-label={presentation.label} sx={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', lineHeight: 0 }}>
      {icon}
    </Box>
  );
  if (!showTooltip) return content;
  return (
    <Tooltip title={presentation.label} arrow>
      {content}
    </Tooltip>
  );
}
