import React from 'react';
import {
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import PublicIcon from '@mui/icons-material/Public';

export default function AddSourcesMenu({
  onUpload,
  onCapturePage,
  disabled = false,
  capturing = false,
  size = 'small',
  tooltip = 'Add source',
}: {
  onUpload: () => void;
  onCapturePage: () => void;
  disabled?: boolean;
  capturing?: boolean;
  size?: 'small' | 'medium' | 'large';
  tooltip?: string;
}) {
  const [anchorEl, setAnchorEl] = React.useState<HTMLElement | null>(null);
  const open = Boolean(anchorEl);

  const close = () => setAnchorEl(null);

  return (
    <>
      <Tooltip title={tooltip}>
        <span>
          <IconButton
            size={size}
            aria-label={tooltip}
            disabled={disabled || capturing}
            onClick={(event) => setAnchorEl(event.currentTarget)}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
      <Menu anchorEl={anchorEl} open={open} onClose={close}>
        <MenuItem
          onClick={() => {
            close();
            onUpload();
          }}
        >
          <ListItemIcon><PictureAsPdfIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Upload PDF</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => {
            close();
            onCapturePage();
          }}
          disabled={capturing}
        >
          <ListItemIcon><PublicIcon fontSize="small" /></ListItemIcon>
          <ListItemText>{capturing ? 'Capturing page…' : 'Capture a page'}</ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
}
