import React from 'react';
import {
  Box,
  ListItem,
  Paper,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { ConversationMarkdown } from './ConversationMarkdown';

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
  rootRef,
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
  rootRef?: React.Ref<HTMLLIElement>;
}) {
  const theme = useTheme();
  const isUser = role === 'user';

  return (
    <ListItem
      ref={rootRef}
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
        <ConversationMarkdown
          content={content}
          sx={{
            cursor: 'text',
            pr: actions ? 2 : 0,
            minWidth: 0,
            maxWidth: '100%',
            '& code': { bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)' },
            '& pre': { bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)' },
          }}
        />
        {afterContent}
      </Paper>
    </ListItem>
  );
});
