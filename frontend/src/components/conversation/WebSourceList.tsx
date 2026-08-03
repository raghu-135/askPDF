import React from 'react';
import { Box, Typography } from '@mui/material';

export type DisplayWebSource = {
  title?: string;
  url?: string;
  text?: string;
  snippet?: string;
};

export function WebSourceList({ sources }: { sources: DisplayWebSource[] }) {
  if (!sources.length) return null;
  return (
    <Box sx={{ mt: 1 }}>
      <details>
        <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
          Web sources used ({sources.length})
        </summary>
        <Box sx={{ mt: 0.75, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
          {sources.map((source, index) => (
            <Box key={`${source.url || source.title}-${index}`} sx={{ p: 1, borderRadius: 1, bgcolor: 'action.hover', borderLeft: '3px solid', borderColor: 'primary.light' }}>
              {source.url ? (
                <Typography variant="caption" component="a" href={source.url} target="_blank" rel="noopener noreferrer" sx={{ color: 'primary.main', display: 'block', fontWeight: 600, textDecoration: 'none', overflowWrap: 'anywhere', '&:hover': { textDecoration: 'underline' } }}>
                  {source.title || source.url}
                </Typography>
              ) : <Typography variant="caption" sx={{ fontWeight: 600 }}>{source.title || 'Web result'}</Typography>}
              {(source.snippet || source.text) && <Typography variant="caption" sx={{ display: 'block', mt: 0.25, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{source.snippet || source.text}</Typography>}
            </Box>
          ))}
        </Box>
      </details>
    </Box>
  );
}
