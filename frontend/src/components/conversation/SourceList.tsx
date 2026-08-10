import React from 'react';
import { Box, Typography } from '@mui/material';

export type DisplaySourceOrigin = {
  run_id: string;
  attempt: number;
  artifact_id: string;
  plan_revision: number;
  inherited: boolean;
};

export type DisplaySource = {
  id?: string;
  kind?: 'web' | 'document' | 'memory' | 'thread';
  title?: string;
  url?: string;
  text?: string;
  snippet?: string;
  origin_attempt?: number;
  inherited?: boolean;
  origins?: DisplaySourceOrigin[];
};

const fallbackTitle = (kind?: DisplaySource['kind']) => ({
  document: 'Document evidence',
  memory: 'Memory',
  thread: 'Thread context',
  web: 'Web result',
}[kind || 'web']);

export function SourceList({ sources, label = 'Sources used' }: { sources: DisplaySource[]; label?: string }) {
  if (!sources.length) return null;
  return (
    <Box sx={{ mt: 1 }}>
      <details>
        <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
          {label} ({sources.length})
        </summary>
        <Box sx={{ mt: 0.75, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
          {sources.map((source, index) => {
            const additionalAttempts = [...new Set((source.origins || [])
              .filter((origin) => origin.attempt && origin.attempt !== source.origin_attempt)
              .map((origin) => origin.attempt))];
            return (
              <Box key={source.id || `${source.url || source.title}-${index}`} sx={{ p: 1, borderRadius: 1, bgcolor: 'action.hover', borderLeft: '3px solid', borderColor: 'primary.light' }}>
                {source.url ? (
                  <Typography variant="caption" component="a" href={source.url} target="_blank" rel="noopener noreferrer" sx={{ color: 'primary.main', display: 'block', fontWeight: 600, textDecoration: 'none', overflowWrap: 'anywhere', '&:hover': { textDecoration: 'underline' } }}>
                    {source.title || source.url}
                  </Typography>
                ) : <Typography variant="caption" sx={{ fontWeight: 600 }}>{source.title || fallbackTitle(source.kind)}</Typography>}
                {(source.snippet || source.text) && <Typography variant="caption" sx={{ display: 'block', mt: 0.25, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{source.snippet || source.text}</Typography>}
                {source.inherited && source.origin_attempt ? <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>From attempt {source.origin_attempt}</Typography> : null}
                {additionalAttempts.length ? <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>Also used in attempt {additionalAttempts.join(', ')}</Typography> : null}
              </Box>
            );
          })}
        </Box>
      </details>
    </Box>
  );
}
