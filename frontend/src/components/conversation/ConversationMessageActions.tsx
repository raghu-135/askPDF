import React from 'react';
import CheckIcon from '@mui/icons-material/Check';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import { IconButton, Tooltip } from '@mui/material';

export function ConversationMessageActions({
  copied,
  readActive = false,
  onCopy,
  onReadAloud,
  children,
}: {
  copied: boolean;
  readActive?: boolean;
  onCopy: () => void;
  onReadAloud?: () => void;
  children?: React.ReactNode;
}) {
  return <>
    <Tooltip title={copied ? 'Copied!' : 'Copy message'}>
      <IconButton
        size="small"
        onClick={onCopy}
        aria-label={copied ? 'Message copied' : 'Copy message'}
        sx={{ color: 'inherit', p: 0.5, '& .MuiSvgIcon-root': { fontSize: '1.1rem' } }}
      >
        {copied ? <CheckIcon fontSize="small" /> : <ContentCopyIcon fontSize="small" />}
      </IconButton>
    </Tooltip>
    {onReadAloud && <Tooltip title="Read aloud">
      <IconButton
        size="small"
        onClick={onReadAloud}
        aria-label="Read aloud"
        color={readActive ? 'primary' : 'inherit'}
        sx={{ p: 0.5, '& .MuiSvgIcon-root': { fontSize: '1.1rem' } }}
      >
        <VolumeUpIcon fontSize="small" />
      </IconButton>
    </Tooltip>}
    {children}
  </>;
}
