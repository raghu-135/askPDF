import React, { useState } from 'react';
import { Box, IconButton, Tooltip, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

export default function InstanceTabCloseLabel({
  title,
  adornment,
  onCopyLink,
  onClose,
  closeTooltip = 'Close tab',
  copyLinkTooltip = 'Copy link',
}: {
  title: string;
  adornment?: React.ReactNode;
  onCopyLink?: () => void | Promise<void>;
  onClose: () => void;
  closeTooltip?: string;
  copyLinkTooltip?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopyLink = async (event: React.MouseEvent) => {
    event.stopPropagation();
    if (!onCopyLink) return;
    try {
      await onCopyLink();
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
      <Typography variant="body2" noWrap sx={{ maxWidth: 170 }}>
        {title}
      </Typography>
      {adornment}
      {onCopyLink ? (
        <Tooltip title={copied ? 'Copied!' : copyLinkTooltip}>
          <IconButton
            size="small"
            onClick={(event) => {
              void handleCopyLink(event);
            }}
            aria-label={copied ? 'Link copied' : copyLinkTooltip}
            sx={{ p: 0.2, flex: '0 0 auto', opacity: 0.72 }}
          >
            {copied ? <CheckIcon sx={{ fontSize: 15 }} /> : <ContentCopyIcon sx={{ fontSize: 15 }} />}
          </IconButton>
        </Tooltip>
      ) : null}
      <Tooltip title={closeTooltip}>
        <IconButton
          size="small"
          onClick={(event) => {
            event.stopPropagation();
            onClose();
          }}
          sx={{ p: 0.2 }}
        >
          <CloseIcon sx={{ fontSize: 15 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
}