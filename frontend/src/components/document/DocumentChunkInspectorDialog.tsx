import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { ChunkIdentityChips, ChunkInspectorBody } from '../inspector';
import { chunkPageLabel } from '../../lib/chunk-page-label';
import {
  getFileChunks,
  type FileChunksResponse,
  type VectorChunk,
} from '../../lib/file-chunks-url';

const PAGE_SIZE = 100;

function chunkMatchesFilter(chunk: VectorChunk, filter: string): boolean {
  const needle = filter.trim().toLowerCase();
  if (!needle) return true;
  const haystacks = [
    String(chunk.chunk_id ?? ''),
    String(chunk.source_id ?? ''),
    chunk.text ?? '',
    chunkPageLabel(chunk),
    JSON.stringify(chunk.metadata ?? {}),
  ];
  return haystacks.some((value) => value.toLowerCase().includes(needle));
}

function ChunkCard({ chunk }: { chunk: VectorChunk }) {
  const copyChunk = async () => {
    try {
      await navigator.clipboard?.writeText(JSON.stringify(chunk, null, 2));
    } catch {
      window.prompt('Copy chunk JSON', JSON.stringify(chunk, null, 2));
    }
  };

  return (
    <Box
      sx={{
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        p: 1.25,
        bgcolor: 'background.paper',
      }}
    >
      <Stack direction="row" alignItems="center" justifyContent="space-between" gap={1} flexWrap="wrap">
        <ChunkIdentityChips chunk={chunk} />
        <Tooltip title="Copy chunk JSON">
          <IconButton size="small" aria-label="Copy chunk JSON" onClick={() => void copyChunk()}>
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Stack>
      <Box sx={{ mt: 1 }}>
        <ChunkInspectorBody
          text={chunk.text ?? ''}
          metadata={chunk.metadata}
          textMaxHeight={180}
          metadataMaxHeight={220}
        />
      </Box>
    </Box>
  );
}

export default function DocumentChunkInspectorDialog({
  open,
  onClose,
  fileHash,
  fileName,
  scope,
  scopeId,
}: {
  open: boolean;
  onClose: () => void;
  fileHash: string;
  fileName: string;
  scope: 'thread' | 'project';
  scopeId: string;
}) {
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<FileChunksResponse | null>(null);

  const loadChunks = useCallback(async (offset: number, append: boolean) => {
    if (append) setLoadingMore(true);
    else setLoading(true);
    setError(null);
    try {
      const next = await getFileChunks(
        scope === 'thread' ? { scope: 'thread', id: scopeId } : { scope: 'project', id: scopeId },
        fileHash,
        { limit: PAGE_SIZE, offset },
      );
      setResponse((prev) => {
        if (!append || !prev) return next;
        return {
          ...next,
          chunks: [...prev.chunks, ...next.chunks],
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      if (!append) setResponse(null);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [fileHash, scope, scopeId]);

  useEffect(() => {
    if (!open) {
      setFilter('');
      setError(null);
      setResponse(null);
      return;
    }
    void loadChunks(0, false);
  }, [open, loadChunks]);

  const filteredChunks = useMemo(
    () => (response?.chunks ?? []).filter((chunk) => chunkMatchesFilter(chunk, filter)),
    [response?.chunks, filter],
  );

  const hasMore = response != null && response.chunks.length < response.total_count;

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md" scroll="paper">
      <DialogTitle sx={{ pr: 6 }}>
        Vector chunks
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          {fileName}
        </Typography>
        <IconButton
          aria-label="Close chunk inspector"
          onClick={onClose}
          sx={{ position: 'absolute', right: 8, top: 8 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {loading ? (
          <Stack direction="row" alignItems="center" gap={1} sx={{ py: 4, justifyContent: 'center' }}>
            <CircularProgress size={22} />
            <Typography variant="body2">Loading indexed chunks…</Typography>
          </Stack>
        ) : error ? (
          <Alert severity="error">{error}</Alert>
        ) : response ? (
          <Stack gap={1.5}>
            <Stack direction="row" alignItems="center" gap={1} flexWrap="wrap">
              <Chip size="small" color="primary" label={`${response.total_count} indexed`} />
              <Chip size="small" variant="outlined" label={response.embedding_model} />
              <Chip size="small" variant="outlined" label={fileHash.slice(0, 12)} />
            </Stack>
            <TextField
              size="small"
              fullWidth
              label="Filter chunks"
              placeholder="Search text, page, chunk id, metadata…"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            />
            {response.total_count === 0 ? (
              <Alert severity="info">No vector chunks indexed for this file yet.</Alert>
            ) : filteredChunks.length === 0 ? (
              <Alert severity="info">No chunks match the current filter.</Alert>
            ) : (
              filteredChunks.map((chunk) => (
                <ChunkCard key={`${chunk.chunk_id}-${chunk.source_id ?? 'chunk'}`} chunk={chunk} />
              ))
            )}
            {hasMore && !filter.trim() ? (
              <Button
                variant="outlined"
                disabled={loadingMore}
                onClick={() => void loadChunks(response.chunks.length, true)}
              >
                {loadingMore ? 'Loading…' : `Load more (${response.chunks.length} of ${response.total_count})`}
              </Button>
            ) : null}
          </Stack>
        ) : null}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
