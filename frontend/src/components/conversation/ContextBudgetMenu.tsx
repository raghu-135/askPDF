import React, { useState } from 'react';
import {
  Box,
  Button,
  Divider,
  Popover,
  Slider,
  TextField,
  Typography,
} from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';
import {
  PACKING_SLICES,
  prefetchSliceSum,
  sliceTokenBudget,
  type ContextBudgetRatios,
} from '../../lib/context-budget';
import { ContextBudgetCapacityBar } from './ContextBudgetCapacityBar';
import { ContextBudgetHelpTip } from './ContextBudgetHelpTip';
import {
  CONTEXT_PACKING_MENU_HELP,
  CONTEXT_SIZE_HELP,
  CONTEXT_WINDOW_SPLIT_HELP,
  SLICE_HELP,
} from '../../lib/context-budget-help';
import { workbenchControlOutlineSx } from '../workbench/WorkbenchToolbar';

type ContextBudgetMenuProps = {
  contextWindow: number;
  contextBudget: ContextBudgetRatios | null;
  defaultContextBudget: ContextBudgetRatios | null;
  disabled?: boolean;
  highlight?: boolean;
  onContextWindowChange: (value: number) => void;
  onContextBudgetChange: (value: ContextBudgetRatios) => void;
  onDismissHelp?: () => void;
};

export function ContextBudgetMenu({
  contextWindow,
  contextBudget,
  defaultContextBudget,
  disabled = false,
  highlight = false,
  onContextWindowChange,
  onContextBudgetChange,
  onDismissHelp,
}: ContextBudgetMenuProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const open = Boolean(anchorEl);

  const handleSliceChange = (key: keyof ContextBudgetRatios, nextValue: number) => {
    if (!contextBudget) return;
    const candidate = { ...contextBudget, [key]: nextValue };
    if (key !== 'usable_ratio' && prefetchSliceSum(candidate) > 1) {
      return;
    }
    onContextBudgetChange(candidate);
  };

  if (!contextBudget || !defaultContextBudget) {
    return (
      <Button
        size="small"
        variant="outlined"
        disabled
        startIcon={<TuneIcon fontSize="small" />}
        sx={{
          minWidth: 116,
          flex: '0 0 auto',
          justifyContent: 'space-between',
          px: 1,
          ...workbenchControlOutlineSx,
        }}
      >
        Ctx {contextWindow > 0 ? contextWindow.toLocaleString() : '—'}
      </Button>
    );
  }

  return (
    <>
      <Button
        size="small"
        variant="outlined"
        disabled={disabled}
        onClick={(event) => {
          onDismissHelp?.();
          setAnchorEl(event.currentTarget);
        }}
        startIcon={<TuneIcon fontSize="small" />}
        sx={{
          minWidth: 116,
          flex: '0 0 auto',
          justifyContent: 'space-between',
          px: 1,
          ...workbenchControlOutlineSx,
          borderColor: highlight ? 'primary.main' : 'divider',
          borderWidth: highlight ? 2 : 1,
        }}
      >
        Ctx {contextWindow > 0 ? contextWindow.toLocaleString() : '—'}
      </Button>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        slotProps={{
          paper: {
            sx: {
              p: 2,
              width: 320,
              maxWidth: '92vw',
              maxHeight: 'min(85vh, 560px)',
              overflowY: 'auto',
            },
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, mb: 1 }}>
          <Typography variant="subtitle2">Context packing budget</Typography>
          <ContextBudgetHelpTip text={CONTEXT_PACKING_MENU_HELP} ariaLabel="Context packing budget help" />
        </Box>
        <TextField
          size="small"
          label="Ctx size (tokens)"
          type="number"
          fullWidth
          value={contextWindow}
          disabled={disabled}
          onChange={(event) => onContextWindowChange(Number.parseInt(event.target.value, 10) || 0)}
          slotProps={{
            htmlInput: { min: 256, step: 1 },
            input: {
              endAdornment: (
                <ContextBudgetHelpTip text={CONTEXT_SIZE_HELP} ariaLabel="Ctx size help" />
              ),
            },
          }}
          sx={{ mb: 2 }}
        />
        {PACKING_SLICES.map(({ key, label }) => {
          const tokens = sliceTokenBudget(contextWindow, contextBudget, key);
          return (
            <Box key={key} sx={{ mb: 1.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, alignItems: 'center' }}>
                <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
                  <Typography variant="caption">{label}</Typography>
                  <ContextBudgetHelpTip text={SLICE_HELP[key]} ariaLabel={`${label} help`} />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {(contextBudget[key] * 100).toFixed(0)}% · {tokens.toLocaleString()} tok
                </Typography>
              </Box>
              <Slider
                size="small"
                min={0.01}
                max={1}
                step={0.01}
                value={contextBudget[key]}
                disabled={disabled}
                onChange={(_, value) => handleSliceChange(key, Array.isArray(value) ? value[0] : value)}
              />
            </Box>
          );
        })}
        <Divider sx={{ my: 1.5 }} />
        <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, mb: 0.5 }}>
          <Typography variant="caption" fontWeight={600}>
            Window split
          </Typography>
          <ContextBudgetHelpTip text={CONTEXT_WINDOW_SPLIT_HELP} ariaLabel="Window split help" />
        </Box>
        <ContextBudgetCapacityBar
          contextWindow={contextWindow}
          contextBudget={contextBudget}
          disabled={disabled}
          showUsableSlider
          onUsableRatioChange={(usableRatio) => onContextBudgetChange({ ...contextBudget, usable_ratio: usableRatio })}
        />
        <Button
          size="small"
          sx={{ mt: 1.5 }}
          onClick={() => onContextBudgetChange(defaultContextBudget)}
          disabled={disabled}
        >
          Reset slice ratios
        </Button>
      </Popover>
    </>
  );
}
