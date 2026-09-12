import React, { useId } from 'react';
import { Box, FormControl, InputLabel, Select, type SxProps, type Theme } from '@mui/material';

function sxList(sx?: SxProps<Theme>) {
  return Array.isArray(sx) ? sx : sx ? [sx] : [];
}

export const workbenchControlOutlineSx = {
  '& fieldset': {
    borderColor: 'transparent',
    borderWidth: '1px',
  },
  '&:hover fieldset': {
    borderColor: 'primary.main',
  },
  '&.Mui-focused fieldset': {
    borderColor: 'primary.main',
  },
};

export function WorkbenchToolbar({
  children,
  trailing,
  sx,
}: {
  children?: React.ReactNode;
  trailing?: React.ReactNode;
  sx?: SxProps<Theme>;
}) {
  return (
    <Box
      sx={[
        {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1,
          flexWrap: 'wrap',
          minWidth: 0,
        },
        ...sxList(sx),
      ]}
    >
      {children != null && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', minWidth: 0, flex: '1 1 auto' }}>
          {children}
        </Box>
      )}
      {trailing}
    </Box>
  );
}

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
          justifyContent: 'flex-end',
          gap: 0.5,
          ml: 'auto',
          flex: '1 1 auto',
          flexWrap: 'wrap',
          minWidth: 0,
        },
        ...sxList(sx),
      ]}
    >
      {children}
    </Box>
  );
}

export function WorkbenchSelect({
  label,
  value,
  disabled = false,
  children,
  minWidth = 220,
  inputLabelId,
  'aria-label': ariaLabel,
  sx,
  onChange,
}: {
  label: string;
  value: string;
  disabled?: boolean;
  children: React.ReactNode;
  minWidth?: number;
  inputLabelId?: string;
  'aria-label'?: string;
  sx?: SxProps<Theme>;
  onChange: (value: string) => void;
}) {
  const generatedId = useId();
  const labelId = inputLabelId || generatedId;
  return (
    <FormControl
      size="small"
      disabled={disabled}
      sx={[{ flex: `0 0 ${minWidth}px`, minWidth }, ...sxList(sx)]}
    >
      <InputLabel id={labelId}>{label}</InputLabel>
      <Select
        labelId={labelId}
        value={value}
        label={label}
        aria-label={ariaLabel}
        onChange={(event) => onChange(String(event.target.value))}
        sx={workbenchControlOutlineSx}
      >
        {children}
      </Select>
    </FormControl>
  );
}
