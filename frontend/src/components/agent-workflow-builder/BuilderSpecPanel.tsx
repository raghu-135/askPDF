import React, { useMemo, useState } from 'react';
import { Box, Button, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import { JsonPreview } from '../agent-graph/AgentGraphInspectorPrimitives';
import type { AgentWorkflowBuilderSpec } from '../../lib/api';

export default function BuilderSpecPanel({ spec }: { spec: AgentWorkflowBuilderSpec }) {
  const [search, setSearch] = useState('');
  const [view, setView] = useState<'structured' | 'raw'>('structured');
  const serialized = useMemo(() => JSON.stringify(spec, null, 2), [spec]);
  const matches = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase();
    if (!needle) return null;
    return serialized.toLocaleLowerCase().split(needle).length - 1;
  }, [search, serialized]);

  const download = () => {
    const blob = new Blob([serialized], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'agent-workflow-spec.json';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box sx={{ height: '100%', minHeight: 0, overflow: 'auto', p: 1.5 }}>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }} sx={{ mb: 1.5 }}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="h6">Workflow specification</Typography>
          <Typography variant="caption" color="text.secondary">Assembled schema sent to validation, save, and test endpoints.</Typography>
        </Box>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={view}
          onChange={(_, next: 'structured' | 'raw' | null) => next && setView(next)}
          aria-label="Specification view"
        >
          <ToggleButton value="structured">Structured</ToggleButton>
          <ToggleButton value="raw">Raw</ToggleButton>
        </ToggleButtonGroup>
        <TextField size="small" label="Search JSON" value={search} onChange={(event) => setSearch(event.target.value)} helperText={matches === null ? ' ' : `${matches} matches`} />
        <Button size="small" startIcon={<ContentCopyIcon />} onClick={() => void navigator.clipboard.writeText(serialized)}>Copy</Button>
        <Button size="small" startIcon={<DownloadIcon />} onClick={download}>Download</Button>
      </Stack>
      {view === 'structured' ? (
        <JsonPreview value={spec} maxHeight={Number.MAX_SAFE_INTEGER} />
      ) : (
        <Box
          component="pre"
          sx={{
            m: 0,
            p: 1.5,
            borderRadius: 1,
            bgcolor: 'background.default',
            color: 'text.primary',
            fontFamily: 'monospace',
            fontSize: 12,
            lineHeight: 1.55,
            overflow: 'auto',
            whiteSpace: 'pre',
          }}
        >
          {serialized}
        </Box>
      )}
    </Box>
  );
}
