import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import WifiTwoToneIcon from '@mui/icons-material/WifiTwoTone';
import WifiOffTwoToneIcon from '@mui/icons-material/WifiOffTwoTone';
import WifiPasswordIcon from '@mui/icons-material/WifiPassword';
import { nextWebSearchMode, type WebSearchMode } from '../../hooks/useWebSearchMode';

const labels: Record<WebSearchMode, string> = {
  off: 'Internet Search Off',
  ask: 'Ask me every time before internet search',
  on: 'Internet Search On',
};

export function WebSearchModeControl({ mode, disabled, onChange }: {
  mode: WebSearchMode;
  disabled?: boolean;
  onChange: (mode: WebSearchMode) => void | Promise<void>;
}) {
  const next = nextWebSearchMode(mode);
  return (
    <Tooltip title={`${labels[mode]}. Click to switch to ${labels[next]}.`} placement="top">
      <span>
        <IconButton
          aria-label={labels[mode]}
          color={mode === 'on' ? 'primary' : mode === 'ask' ? 'warning' : 'default'}
          onClick={() => void onChange(next)}
          disabled={disabled}
          size="small"
          sx={{ p: 0.5 }}
        >
          {mode === 'on' ? <WifiTwoToneIcon /> : mode === 'ask' ? <WifiPasswordIcon /> : <WifiOffTwoToneIcon />}
        </IconButton>
      </span>
    </Tooltip>
  );
}
