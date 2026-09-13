import React from 'react';
import { Typography, type SxProps, type Theme } from '@mui/material';
import dynamic from 'next/dynamic';
import remarkGfm from 'remark-gfm';

const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });

const MarkdownContent = React.memo(function MarkdownContent({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
});

export const conversationMarkdownSx: SxProps<Theme> = {
  overflowWrap: 'anywhere',
  wordBreak: 'break-word',
  '& p': { m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& p:last-child': { mb: 0 },
  '& ul, & ol': { pl: 2, m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& li': { mb: 0.5, overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& blockquote': { m: 0, pl: 1.5, borderLeft: '3px solid', borderColor: 'divider', overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& a': { overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& code': { bgcolor: 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace', overflowWrap: 'anywhere', wordBreak: 'break-word' },
  '& pre': { maxWidth: '100%', bgcolor: 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 },
  '& pre code': { overflowWrap: 'normal', wordBreak: 'normal' },
  '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto', borderCollapse: 'collapse', mb: 1 },
  '& th, & td': { border: '1px solid', borderColor: 'divider', px: 0.75, py: 0.5 },
};

export function ConversationMarkdown({
  content,
  sx,
}: {
  content: string;
  sx?: SxProps<Theme>;
}) {
  return (
    <Typography variant="body2" component="div" sx={[conversationMarkdownSx, ...(Array.isArray(sx) ? sx : sx ? [sx] : [])]}>
      <MarkdownContent content={content} />
    </Typography>
  );
}
