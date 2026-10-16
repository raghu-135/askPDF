import React from 'react';
import { Chip, Stack } from '@mui/material';
import { chunkPageLabel, type ChunkPageFields } from '../../lib/chunk-page-label';

export type ChunkIdentityFields = ChunkPageFields & {
  chunk_id?: number | string | null;
  section_id?: string | null;
  table_id?: string | null;
  file_name?: string | null;
  file_hash?: string | null;
};

export function ChunkIdentityChips({
  chunk,
  showFileName = false,
}: {
  chunk: ChunkIdentityFields;
  showFileName?: boolean;
}) {
  const page = chunkPageLabel(chunk);

  return (
    <Stack direction="row" spacing={0.75} alignItems="center" useFlexGap flexWrap="wrap" sx={{ gap: 0.75 }}>
      <Chip size="small" variant="outlined" label={`chunk ${chunk.chunk_id ?? '?'}`} />
      {page ? <Chip size="small" variant="outlined" label={page} /> : null}
      {showFileName && (chunk.file_name || chunk.file_hash) ? (
        <Chip size="small" label={chunk.file_name || chunk.file_hash!.slice(0, 10)} />
      ) : null}
      {chunk.table_id ? (
        <Chip size="small" variant="outlined" color="warning" label={`table ${chunk.table_id}`} />
      ) : null}
      {chunk.section_id ? (
        <Chip size="small" variant="outlined" color="secondary" label={`section ${chunk.section_id}`} />
      ) : null}
    </Stack>
  );
}
