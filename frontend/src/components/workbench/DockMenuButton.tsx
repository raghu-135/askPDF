import React, { useState } from 'react';
import {
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import KeyboardDoubleArrowDownIcon from '@mui/icons-material/KeyboardDoubleArrowDown';
import KeyboardDoubleArrowLeftIcon from '@mui/icons-material/KeyboardDoubleArrowLeft';
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import type {
  ResolvedWorkbenchPlacement,
  WorkbenchLayoutState,
  WorkbenchPlacement,
} from '../../lib/workbench-layout';

const placementIcon = (placement: WorkbenchPlacement | ResolvedWorkbenchPlacement) => {
  if (placement === 'left') return <KeyboardDoubleArrowLeftIcon fontSize="small" />;
  if (placement === 'bottom') return <KeyboardDoubleArrowDownIcon fontSize="small" />;
  if (placement === 'auto') return <AutoAwesomeIcon fontSize="small" />;
  return <KeyboardDoubleArrowRightIcon fontSize="small" />;
};

const labels: Record<WorkbenchPlacement, string> = {
  auto: 'Auto position',
  left: 'Dock left',
  right: 'Dock right',
  bottom: 'Dock bottom',
};

export default function DockMenuButton({
  value,
  resolvedPlacement,
  onChange,
  label = 'Panel layout',
}: {
  value: WorkbenchLayoutState;
  resolvedPlacement: ResolvedWorkbenchPlacement;
  onChange: (layout: WorkbenchLayoutState) => void;
  label?: string;
}) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  return (
    <>
      <Tooltip title={`${label}: ${value.visible ? labels[value.placement] : 'Hidden'}`}>
        <IconButton
          color="primary"
          size="small"
          aria-label={label}
          aria-controls={anchorEl ? 'workbench-dock-menu' : undefined}
          aria-haspopup="menu"
          aria-expanded={Boolean(anchorEl)}
          onClick={(event) => setAnchorEl(event.currentTarget)}
        >
          {value.visible ? placementIcon(value.placement === 'auto' ? 'auto' : resolvedPlacement) : <VisibilityOffIcon fontSize="small" />}
        </IconButton>
      </Tooltip>
      <Menu
        id="workbench-dock-menu"
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        {(Object.keys(labels) as WorkbenchPlacement[]).map((placement) => (
          <MenuItem
            key={placement}
            selected={value.visible && value.placement === placement}
            onClick={() => {
              onChange({ ...value, placement, visible: true });
              setAnchorEl(null);
            }}
          >
            <ListItemIcon>{placementIcon(placement)}</ListItemIcon>
            <ListItemText>{labels[placement]}</ListItemText>
          </MenuItem>
        ))}
        <MenuItem
          selected={!value.visible}
          onClick={() => {
            onChange({ ...value, visible: !value.visible });
            setAnchorEl(null);
          }}
        >
          <ListItemIcon>{value.visible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}</ListItemIcon>
          <ListItemText>{value.visible ? 'Hide panel' : 'Show panel'}</ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
}
