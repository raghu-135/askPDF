import React, { useMemo } from 'react';
import { Chip, Stack } from '@mui/material';
import type { CitationDisplaySource } from '../../lib/research-citation-display';
import { ConversationMarkdown } from './ConversationMarkdown';

const CITATION_CHIP_PATTERN = /([^\n`]+?,\s*page\s+\d+)\s*\(`(doc:[^`]+)`\)/gi;

export function ConversationMarkdownWithCitations({
  content,
  citationSources,
}: {
  content: string;
  citationSources?: CitationDisplaySource[];
}) {
  const segments = useMemo(() => {
    const text = String(content || '');
    if (!text.trim()) return [{ type: 'markdown' as const, value: text }];
    const parts: Array<{ type: 'markdown' | 'chip'; value: string; sourceId?: string }> = [];
    let lastIndex = 0;
    const pattern = new RegExp(CITATION_CHIP_PATTERN.source, 'gi');
    let match = pattern.exec(text);
    while (match) {
      if (match.index > lastIndex) {
        parts.push({ type: 'markdown', value: text.slice(lastIndex, match.index) });
      }
      parts.push({ type: 'chip', value: match[1].trim(), sourceId: match[2] });
      lastIndex = match.index + match[0].length;
      match = pattern.exec(text);
    }
    if (lastIndex < text.length) {
      parts.push({ type: 'markdown', value: text.slice(lastIndex) });
    }
    return parts.length ? parts : [{ type: 'markdown' as const, value: text }];
  }, [content]);

  if (segments.length === 1 && segments[0].type === 'markdown') {
    return <ConversationMarkdown content={segments[0].value} />;
  }

  return (
    <Stack spacing={0.5}>
      {segments.map((segment, index) => {
        if (segment.type === 'chip') {
          return (
            <Chip
              key={`cite-${index}-${segment.sourceId}`}
              size="small"
              variant="outlined"
              label={segment.value}
              title={segment.sourceId}
              sx={{ alignSelf: 'flex-start', maxWidth: '100%', height: 'auto', '& .MuiChip-label': { whiteSpace: 'normal' } }}
            />
          );
        }
        if (!segment.value.trim()) return null;
        return <ConversationMarkdown key={`md-${index}`} content={segment.value} />;
      })}
    </Stack>
  );
}
