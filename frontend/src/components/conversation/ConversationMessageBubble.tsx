import React from 'react';
import {
  Box,
  ListItem,
  Paper,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import dynamic from 'next/dynamic';
import remarkGfm from 'remark-gfm';

const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });

const MemoizedMarkdown = React.memo(function MemoizedMarkdown({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
});

export type ConversationMessageRole = 'user' | 'assistant';

export const ConversationMessageBubble = React.memo(function ConversationMessageBubble({
  role,
  content,
  active = false,
  recollected = false,
  editing = false,
  wide = false,
  badge,
  actions,
  beforeContent,
  afterContent,
}: {
  role: ConversationMessageRole;
  content: string;
  active?: boolean;
  recollected?: boolean;
  editing?: boolean;
  wide?: boolean;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  beforeContent?: React.ReactNode;
  afterContent?: React.ReactNode;
}) {
  const theme = useTheme();
  const isUser = role === 'user';

  return (
    <ListItem
      alignItems="flex-start"
      sx={{
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        px: 0,
        py: 0.5,
      }}
    >
      <Paper
        sx={{
          p: 1.5,
          bgcolor: isUser
            ? theme.palette.mode === 'dark'
              ? theme.palette.primary.dark
              : theme.palette.primary.main
            : theme.palette.mode === 'dark'
              ? theme.palette.background.paper
              : theme.palette.grey[100],
          color: isUser
            ? theme.palette.getContrastText(theme.palette.primary.main)
            : theme.palette.text.primary,
          width: wide ? `calc(100% - ${theme.spacing(6)})` : 'fit-content',
          maxWidth: isUser ? '90%' : `calc(100% - ${theme.spacing(6)})`,
          minWidth: 0,
          overflowWrap: 'anywhere',
          wordBreak: 'break-word',
          boxShadow: active
            ? '0 0 10px rgba(255, 255, 0, 0.4)'
            : recollected
              ? '0 0 10px rgba(156, 39, 176, 0.5)'
              : 'none',
          border: recollected || editing ? '2px solid' : 'none',
          borderColor: editing
            ? 'warning.main'
            : recollected
              ? 'secondary.main'
              : 'transparent',
          borderRadius: '12px',
          transition: 'all 0.2s ease',
          cursor: 'default',
          position: 'relative',
          contain: 'layout paint style',
          '&:hover .message-actions': {
            opacity: 1,
          },
        }}
      >
        {badge}
        {actions && (
          <Box
            className="message-actions"
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              display: 'flex',
              gap: 0.25,
              opacity: 0,
              transition: 'opacity 0.2s ease',
              bgcolor: isUser
                ? theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.2)'
                : theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
              backdropFilter: 'blur(4px)',
              borderRadius: '20px',
              p: 0.4,
              boxShadow: 1,
              zIndex: 10,
              '&:hover': { opacity: 1 },
            }}
          >
            {actions}
          </Box>
        )}
        {beforeContent}
        <Typography variant="body2" component="div" sx={{
          cursor: 'text',
          pr: actions ? 2 : 0,
          minWidth: 0,
          maxWidth: '100%',
          overflowWrap: 'anywhere',
          wordBreak: 'break-word',
          '& p': { m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& p:last-child': { mb: 0 },
          '& ul, & ol': { pl: 2, m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& li': { mb: 0.5, overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& blockquote': { m: 0, pl: 1.5, borderLeft: '3px solid', borderColor: 'divider', overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& a': { overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& code': { bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace', overflowWrap: 'anywhere', wordBreak: 'break-word' },
          '& pre': { maxWidth: '100%', bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 },
          '& pre code': { overflowWrap: 'normal', wordBreak: 'normal' },
          '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto', borderCollapse: 'collapse', mb: 1 },
          '& th, & td': { border: '1px solid', borderColor: 'divider', px: 0.75, py: 0.5 },
        }}>
          <MemoizedMarkdown content={content} />
        </Typography>
        {afterContent}
      </Paper>
    </ListItem>
  );
});
