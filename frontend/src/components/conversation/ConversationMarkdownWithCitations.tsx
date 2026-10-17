import React, { useMemo } from 'react';
import { Chip, Stack } from '@mui/material';
import type { CitationDisplaySource } from '../../lib/research-citation-display';
import { ConversationMarkdown } from './ConversationMarkdown';

const DOC_CITATION_PATTERN = /([^\n`]+?,\s*page\s+\d+)\s*\(`(doc:[^`]+)`\)/gi;
const SOURCE_BRACKET_PATTERN = /\[Source:\s*([^\]]+)\]/gi;

type Segment = { type: 'markdown' | 'chip'; value: string; sourceId?: string };

function collectCitationSegments(text: string): Segment[] {
  const hits: Array<{ index: number; length: number; segment: Segment }> = [];
  const docPattern = new RegExp(DOC_CITATION_PATTERN.source, 'gi');
  let match = docPattern.exec(text);
  while (match) {
    hits.push({
      index: match.index,
      length: match[0].length,
      segment: { type: 'chip', value: match[1].trim(), sourceId: match[2] },
    });
    match = docPattern.exec(text);
  }
  const sourcePattern = new RegExp(SOURCE_BRACKET_PATTERN.source, 'gi');
  match = sourcePattern.exec(text);
  while (match) {
    const overlaps = hits.some((hit) => match!.index >= hit.index && match!.index < hit.index + hit.length);
    if (!overlaps) {
      hits.push({
        index: match.index,
        length: match[0].length,
        segment: { type: 'chip', value: match[1].trim(), sourceId: match[0] },
      });
    }
    match = sourcePattern.exec(text);
  }
  hits.sort((a, b) => a.index - b.index);
  const parts: Segment[] = [];
  let lastIndex = 0;
  hits.forEach((hit) => {
    if (hit.index > lastIndex) {
      parts.push({ type: 'markdown', value: text.slice(lastIndex, hit.index) });
    }
    parts.push(hit.segment);
    lastIndex = hit.index + hit.length;
  });
  if (lastIndex < text.length) {
    parts.push({ type: 'markdown', value: text.slice(lastIndex) });
  }
  return parts.length ? parts : [{ type: 'markdown', value: text }];
}

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
    return collectCitationSegments(text);
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
