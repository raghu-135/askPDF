import React from 'react';
import { Box, Typography } from '@mui/material';
import type { Thread } from '../../lib/api';
import ThreadReferenceChip from '../ThreadReferenceChip';
import { OverviewSeparatedItem } from './OverviewSection';

export type ThreadStatsDocument = {
  id: string;
  fileName?: string | null;
  pageCount?: number | string | null;
  addedAt?: string | null;
  onSelect?: () => void;
};

export type ThreadStatsListProps = {
  thread: Thread;
  projectName?: string | null;
  threadsById?: Map<string, Thread>;
  onOpenThread?: (thread: Thread) => void;
  documents?: ThreadStatsDocument[];
  compact?: boolean;
  showMessageCounts?: boolean;
};

export default function ThreadStatsList({
  thread,
  projectName,
  threadsById = new Map(),
  onOpenThread,
  documents = [],
  compact = false,
  showMessageCounts = true,
}: ThreadStatsListProps) {
  const forkInfo = thread.thread_metadata?.fork;
  const childThreadIds = Array.isArray(thread.thread_metadata?.fork_children)
    ? thread.thread_metadata.fork_children.filter((id): id is string => typeof id === 'string' && id.length > 0)
    : [];
  const forkSummary = forkInfo?.forked_at
    ? `Forked ${forkInfo.mode === 'from_message' ? 'from a message' : 'from full thread'} on ${new Date(forkInfo.forked_at).toLocaleString()}`
    : null;
  const labelVariant = compact ? 'caption' : 'body2';
  const valueVariant = compact ? 'caption' : 'body2';

  const renderValue = (value: React.ReactNode) => (
    <Typography variant={valueVariant} component="div" sx={{ wordBreak: 'break-word' }}>
      {value}
    </Typography>
  );

  return (
    <Box
      sx={{
        ...(compact ? { p: 0.5, pr: 0.75, minWidth: 220, maxWidth: 320 } : {}),
        ...(compact ? { maxHeight: 'min(360px, calc(100vh - 96px))', overflowY: 'auto' } : {}),
      }}
      onClick={compact ? (event) => event.stopPropagation() : undefined}
    >
      {projectName ? (
        <OverviewSeparatedItem label="Project">
          {renderValue(projectName)}
        </OverviewSeparatedItem>
      ) : null}
      <OverviewSeparatedItem label="Created">
        {renderValue(new Date(thread.created_at).toLocaleString())}
      </OverviewSeparatedItem>
      {thread.last_activity_at ? (
        <OverviewSeparatedItem label="Last activity">
          {renderValue(new Date(thread.last_activity_at).toLocaleString())}
        </OverviewSeparatedItem>
      ) : null}
      <OverviewSeparatedItem label="Embedding model">
        {renderValue(thread.embeddingModel)}
      </OverviewSeparatedItem>
      {showMessageCounts && thread.message_count !== undefined ? (
        <OverviewSeparatedItem label="Messages">
          {renderValue(thread.message_count)}
        </OverviewSeparatedItem>
      ) : null}
      {showMessageCounts && thread.file_count !== undefined ? (
        <OverviewSeparatedItem label="Files">
          {renderValue(thread.file_count)}
        </OverviewSeparatedItem>
      ) : null}
      {forkSummary ? (
        <OverviewSeparatedItem label="Lineage">
          {renderValue(forkSummary)}
        </OverviewSeparatedItem>
      ) : null}
      {forkInfo?.parent_thread_id ? (
        <OverviewSeparatedItem label="Parent thread">
          <ThreadReferenceChip
            threadId={forkInfo.parent_thread_id}
            fallbackName={forkInfo.parent_thread_name || 'deleted thread'}
            threadsById={threadsById}
            onOpenThread={onOpenThread}
          />
        </OverviewSeparatedItem>
      ) : null}
      {forkInfo?.memory_copy_mode ? (
        <OverviewSeparatedItem label="Memory copy">
          {renderValue(
            `${forkInfo.memory_copy_mode.replace(/_/g, ' ')}${Array.isArray(forkInfo.copied_memory_ids) ? ` (${forkInfo.copied_memory_ids.length})` : ''}`,
          )}
        </OverviewSeparatedItem>
      ) : null}
      <OverviewSeparatedItem label="Child threads">
        {childThreadIds.length > 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
            {childThreadIds.map((childId) => (
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
          <Typography variant={labelVariant} color="text.secondary" component="div">
            No child threads
          </Typography>
        )}
      </OverviewSeparatedItem>
      {documents.length > 0 ? (
        <OverviewSeparatedItem label="Documents">
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
            {documents.map((document) => (
              <Box
                key={document.id}
                sx={{
                  minWidth: 0,
                  ...(document.onSelect ? {
                    cursor: 'pointer',
                    borderRadius: 1,
                    px: 0.5,
                    mx: -0.5,
                    '&:hover': { bgcolor: 'action.hover' },
                  } : {}),
                }}
                onClick={document.onSelect}
                role={document.onSelect ? 'button' : undefined}
                tabIndex={document.onSelect ? 0 : undefined}
                onKeyDown={document.onSelect ? (event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    document.onSelect?.();
                  }
                } : undefined}
              >
                {document.fileName ? (
                  <Typography
                    variant={compact ? 'caption' : 'body2'}
                    component="div"
                    sx={{ fontWeight: 600, lineHeight: 1.25, wordBreak: 'break-word' }}
                  >
                    {document.fileName}
                  </Typography>
                ) : null}
                {document.pageCount !== undefined && document.pageCount !== null && document.pageCount !== '' ? (
                  <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                    Pages: {document.pageCount}
                  </Typography>
                ) : null}
                {document.addedAt ? (
                  <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                    Added: {new Date(document.addedAt).toLocaleString()}
                  </Typography>
                ) : null}
              </Box>
            ))}
          </Box>
        </OverviewSeparatedItem>
      ) : null}
    </Box>
  );
}

export const threadDocumentsFromMeta = (thread: Thread): ThreadStatsDocument[] => (
  Object.entries(thread.documents_meta || {})
    .filter((entry): entry is [string, NonNullable<Thread['documents_meta']>[string]] => {
      const meta = entry[1];
      return !!meta && typeof meta === 'object' && !Array.isArray(meta);
    })
    .filter(([, meta]) => Boolean(meta.file_name || meta.page_count || meta.document_available_in_thread_at))
    .map(([fileHash, meta]) => ({
      id: fileHash,
      fileName: meta.file_name,
      pageCount: meta.page_count,
      addedAt: meta.document_available_in_thread_at,
    }))
);
