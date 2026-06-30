import React from 'react';
import { Chip, Typography } from '@mui/material';
import { SxProps, Theme } from '@mui/material/styles';
import { Thread } from '../lib/api';

interface ThreadReferenceChipProps {
  threadId: string | null | undefined;
  fallbackName?: string | null;
  threadsById: Map<string, Thread>;
  onOpenThread?: (thread: Thread) => void;
  sx?: SxProps<Theme>;
}

const ThreadReferenceChip: React.FC<ThreadReferenceChipProps> = ({
  threadId,
  fallbackName,
  threadsById,
  onOpenThread,
  sx,
}) => {
  const chipSx: SxProps<Theme> = {
    height: 22,
    maxWidth: '100%',
    color: 'primary.contrastText',
    bgcolor: 'primary.main',
    justifyContent: 'flex-start',
    '& .MuiChip-label': {
      px: 0.75,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    },
    '&:hover, &:focus-visible': {
      bgcolor: 'primary.dark',
    },
  };

  if (!threadId) {
    return <Typography variant="caption" color="text.secondary">None</Typography>;
  }

  const target = threadsById.get(threadId);
  if (!target) {
    return (
      <Typography variant="caption" color="text.secondary">
        {fallbackName || threadId}
      </Typography>
    );
  }

  return (
    <Chip
      label={target.name}
      size="small"
      clickable={Boolean(onOpenThread)}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onOpenThread?.(target);
      }}
      onMouseDown={(event) => event.stopPropagation()}
      sx={[
        chipSx,
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
    />
  );
};

export default ThreadReferenceChip;
