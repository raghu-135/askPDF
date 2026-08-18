import React from 'react';
import DownloadIcon from '@mui/icons-material/Download';
import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material';
import { ConversationDisclosure } from './ConversationDisclosure';

export type DisplayConversationArtifact = {
  id: string;
  kind: string;
  byte_size: number;
  media_type?: string;
};

export function ConversationArtifactList({
  artifacts,
  onDownload,
}: {
  artifacts: DisplayConversationArtifact[];
  onDownload: (artifact: DisplayConversationArtifact) => void;
}) {
  if (!artifacts.length) return null;
  return (
    <ConversationDisclosure label={`Artifacts (${artifacts.length})`}>
      <Stack spacing={0.5}>
        {artifacts.map((artifact) => {
          const label = artifact.kind.replaceAll('_', ' ');
          return <Box key={artifact.id} sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1, py: 0.5, borderRadius: 1, bgcolor: 'action.hover' }}>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              <Typography variant="caption" fontWeight={600} sx={{ display: 'block' }}>{label}</Typography>
              <Typography variant="caption" color="text.secondary">
                {Math.max(1, Math.ceil(artifact.byte_size / 1024))} KB{artifact.media_type ? ` · ${artifact.media_type}` : ''}
              </Typography>
            </Box>
            <Tooltip title={`Download ${label}`}>
              <IconButton size="small" onClick={() => onDownload(artifact)} aria-label={`Download ${label}`}>
                <DownloadIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          </Box>;
        })}
      </Stack>
    </ConversationDisclosure>
  );
}
