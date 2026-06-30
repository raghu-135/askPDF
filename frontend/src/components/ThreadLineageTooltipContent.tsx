import React from 'react';
import { Box, Typography } from '@mui/material';
import { Thread } from '../lib/api';
import ThreadReferenceChip from './ThreadReferenceChip';

interface ThreadLineageTooltipContentProps {
  thread: Thread;
  threadsById: Map<string, Thread>;
  onOpenThread?: (thread: Thread) => void;
}

const ThreadLineageTooltipContent: React.FC<ThreadLineageTooltipContentProps> = ({
  thread,
  threadsById,
  onOpenThread,
}) => {
  const forkInfo = thread.thread_metadata?.fork;
  const childThreadIds = Array.isArray(thread.thread_metadata?.fork_children)
    ? thread.thread_metadata.fork_children.filter((id): id is string => typeof id === 'string' && id.length > 0)
    : [];
  const forkSummary = forkInfo?.forked_at
    ? `Forked ${forkInfo.mode === 'from_message' ? 'from a message' : 'from full thread'} on ${new Date(forkInfo.forked_at).toLocaleString()}`
    : 'Thread lineage';
  const documents = Object.entries(thread.documents_meta || {})
    .filter((entry): entry is [string, NonNullable<Thread['documents_meta']>[string]] => {
      const meta = entry[1];
      return !!meta && typeof meta === 'object' && !Array.isArray(meta);
    })
    .filter(([, meta]) =>
      Boolean(meta.file_name || meta.page_count || meta.document_available_in_thread_at)
    );
  const sectionSx = {
    pt: 0.75,
    mt: 0.75,
    borderTop: 1,
    borderColor: 'divider',
    '&:first-of-type': {
      pt: 0,
      mt: 0,
      borderTop: 0,
    },
  };

  return (
    <Box
      sx={{
        p: 0.5,
        pr: 0.75,
        minWidth: 220,
        maxWidth: 320,
      }}
      onClick={(event) => event.stopPropagation()}
    >
      <Box sx={sectionSx}>
        <Typography variant="caption" color="text.secondary" component="div">
          Created
        </Typography>
        <Typography variant="caption" component="div">
          {new Date(thread.created_at).toLocaleString()}
        </Typography>
      </Box>
      <Box sx={sectionSx}>
        <Typography variant="caption" color="text.secondary" component="div">
          Embedding model
        </Typography>
        <Typography variant="caption" component="div" sx={{ wordBreak: 'break-word' }}>
          {thread.embed_model}
        </Typography>
      </Box>
      {(forkInfo || childThreadIds.length > 0) && (
        <Box sx={sectionSx}>
          <Typography variant="caption" color="text.secondary" component="div">
            {forkSummary}
          </Typography>
        </Box>
      )}
      {forkInfo?.parent_thread_id && (
        <Box sx={sectionSx}>
          <Typography variant="caption" color="text.secondary" component="div">
            Parent thread
          </Typography>
          <ThreadReferenceChip
            threadId={forkInfo.parent_thread_id}
            fallbackName={forkInfo.parent_thread_name || 'deleted thread'}
            threadsById={threadsById}
            onOpenThread={onOpenThread}
          />
        </Box>
      )}
      <Box sx={sectionSx}>
        <Typography variant="caption" color="text.secondary" component="div">
          Child threads
        </Typography>
        {childThreadIds.length > 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
            {childThreadIds.map(childId => (
              <Box key={childId}>
                <ThreadReferenceChip
                  threadId={childId}
                  threadsById={threadsById}
                  onOpenThread={onOpenThread}
                />
              </Box>
            ))}
          </Box>
        ) : (
          <Typography variant="caption" color="text.secondary" component="div">
            No child threads
          </Typography>
        )}
      </Box>
      {documents.length > 0 && (
        <Box sx={sectionSx}>
          <Typography variant="caption" color="text.secondary" component="div">
            Documents
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
            {documents.map(([fileHash, meta]) => (
              <Box key={fileHash} sx={{ minWidth: 0 }}>
                {meta.file_name && (
                  <Typography variant="caption" component="div" sx={{ fontWeight: 600, lineHeight: 1.25, wordBreak: 'break-word' }}>
                    {meta.file_name}
                  </Typography>
                )}
                {meta.page_count !== undefined && meta.page_count !== null && meta.page_count !== '' && (
                  <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                    Pages: {meta.page_count}
                  </Typography>
                )}
                {meta.document_available_in_thread_at && (
                  <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                    Added: {new Date(meta.document_available_in_thread_at).toLocaleString()}
                  </Typography>
                )}
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default ThreadLineageTooltipContent;
