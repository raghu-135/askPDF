import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import LanguageTwoToneIcon from '@mui/icons-material/LanguageTwoTone';
import LanguageOutlinedIcon from '@mui/icons-material/LanguageOutlined';
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import { nextBrowserAccessMode, type BrowserAccessMode } from '../../hooks/useBrowserAccessMode';

const labels: Record<BrowserAccessMode, string> = {
  off: 'Browser agent off',
  ask: 'Ask before the agent uses the workbench browser',
  on: 'Browser agent on',
};

export function BrowserAccessModeControl({ mode, disabled, onChange }: {
  mode: BrowserAccessMode;
  disabled?: boolean;
  onChange: (mode: BrowserAccessMode) => void | Promise<void>;
}) {
  const next = nextBrowserAccessMode(mode);
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
          {mode === 'on' ? <LanguageTwoToneIcon /> : mode === 'ask' ? <TravelExploreIcon /> : <LanguageOutlinedIcon />}
        </IconButton>
      </span>
    </Tooltip>
  );
}